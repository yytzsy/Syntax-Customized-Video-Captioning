import random

import torch
import torch.nn as nn
import torch.nn.functional as F

from .Attention import Attention
from ON_LSTM import ONLSTMStack
from PLAIN_ON_LSTM import PLAIN_ONLSTMStack
import numpy as np


class DecoderRNN(nn.Module):
    """
    Provides functionality for decoding in a seq2seq framework, with an option for attention.
    Args:
        vocab_size (int): size of the vocabulary
        max_len (int): a maximum allowed length for the sequence to be processed
        dim_hidden (int): the number of features in the hidden state `h`
        n_layers (int, optional): number of recurrent layers (default: 1)
        rnn_cell (str, optional): type of RNN cell (default: gru)
        bidirectional (bool, optional): if the encoder is bidirectional (default False)
        input_dropout_p (float, optional): dropout probability for the input sequence (default: 0)
        rnn_dropout_p (float, optional): dropout probability for the output sequence (default: 0)

    """

    def __init__(self,
                 vocab_size,
                 max_len,
                 dim_hidden,
                 dim_word,
                 dim_parse,
                 n_layers=1,
                 bidirectional=False,
                 input_dropout_p=0.1,
                 rnn_dropout_p=0.1,
                 w_dropout_p = 0.4,
                 chunk_size = 10,
                 embedding_pretrained_weights = None):
        super(DecoderRNN, self).__init__()

        self.bidirectional_encoder = bidirectional
        self.dim_output = vocab_size
        self.dim_hidden = dim_hidden * 2 if bidirectional else dim_hidden
        self.dim_word = dim_word
        self.dim_parse = dim_parse
        self.max_length = max_len
        self.sos_id = 1
        self.eos_id = 0
        self.input_dropout_input = nn.Dropout(input_dropout_p)
        self.input_dropout_control = nn.Dropout(input_dropout_p)

        self.attention_video = Attention(self.dim_hidden,self.dim_hidden,n_layers)
        self.attention_parse = Attention(self.dim_hidden,self.dim_hidden,n_layers)

        self.embedding = nn.Embedding(self.dim_output, dim_word)
        if embedding_pretrained_weights is not None:
            self.embedding.weight = nn.Parameter(torch.FloatTensor(embedding_pretrained_weights))

        self.out = nn.Linear(self.dim_hidden, self.dim_output)

        self.rnn_parse = ONLSTMStack(
            [self.dim_word+self.dim_hidden] + [self.dim_hidden] * n_layers,
            dim_hidden,
            dropconnect=w_dropout_p,
            dropout=rnn_dropout_p
        )

        self.rnn_word = ONLSTMStack(
            [self.dim_hidden*2] + [self.dim_hidden] * n_layers,
            dim_hidden,
            dropconnect=w_dropout_p,
            dropout=rnn_dropout_p
        )

        self._init_weights()

    def forward(self,
                encoder_outputs,
                encoder_outputs_parse,
                targets=None,
                mode='train',
                opt={},
                template = 0):
        """

        Inputs: inputs, encoder_hidden, encoder_outputs, function, teacher_forcing_ratio
        - **encoder_hidden** (num_layers * num_directions, batch_size, dim_hidden): tensor containing the features in the
          hidden state `h` of encoder. Used as the initial hidden state of the decoder. (default `None`)
        - **encoder_outputs** (batch, seq_len, dim_hidden * num_directions): (default is `None`).
        - **targets** (batch, max_length): targets labels of the ground truth sentences

        Outputs: seq_probs,
        - **seq_logprobs** (batch_size, max_length, vocab_size): tensors containing the outputs of the decoding function.
        - **seq_preds** (batch_size, max_length): predicted symbols
        """


        sample_max = opt.get('sample_max', 1)
        beam_size = opt.get('beam_size', 1)
        temperature = opt.get('temperature', 1.0)

        batch_size, _, _ = encoder_outputs.size()

        seq_logprobs = []
        seq_preds = []
        hidden_parse_feats_list = []
        hidden_feats_list = []

        decoder_hidden_parse = self.init_parse_hidden(batch_size)
        decoder_hidden = self.init_word_hidden(batch_size)


        if mode == 'train':

            targets_emb = self.embedding(targets)
            for i in range(self.max_length - 1):

                current_words = targets_emb[:, i, :]
                prev_state_parse = list(decoder_hidden_parse)
                prev_state = list(decoder_hidden)
                if len(prev_state_parse)==1:
                    attention_hidden_parse, attention_cell_parse = prev_state_parse[0]
                    attention_hidden, attention_cell = prev_state[0]
                else:
                    for jj in range(len(prev_state_parse)):
                        hidden_item_parse, cell_item_parse = prev_state_parse[jj]
                        if jj > 0:
                            attention_hidden_parse = torch.cat((attention_hidden_parse,hidden_item_parse),-1)
                            attention_cell_parse = torch.cat((attention_cell_parse,cell_item_parse),-1)
                        else:
                            attention_hidden_parse = hidden_item_parse
                            attention_cell_parse = cell_item_parse
                    for jj in range(len(prev_state)):
                        hidden_item, cell_item = prev_state[jj]
                        if jj > 0:
                            attention_hidden = torch.cat((attention_hidden,hidden_item),-1)
                            attention_cell = torch.cat((attention_cell,cell_item),-1)
                        else:
                            attention_hidden = hidden_item
                            attention_cell = cell_item
                context_parse = self.attention_parse(attention_hidden_parse, encoder_outputs_parse)
                context_video = self.attention_video(attention_hidden, encoder_outputs)

                decoder_input_parse = torch.cat([current_words,context_parse], dim=1)
                decoder_input_parse = self.input_dropout_input(decoder_input_parse).unsqueeze(1)
                decoder_input_parse = decoder_input_parse.permute(1,0,2)
                control_input_parse = self.input_dropout_input(context_parse).unsqueeze(1)
                control_input_parse = control_input_parse.permute(1,0,2)
                decoder_output_parse, decoder_hidden_parse, _, _ = self.rnn_parse(decoder_input_parse, decoder_hidden_parse, control_input_parse)
                decoder_output_parse = torch.squeeze(decoder_output_parse)

                decoder_input = torch.cat([decoder_output_parse,context_video], dim=1)
                decoder_input = self.input_dropout_input(decoder_input).unsqueeze(1)
                decoder_input = decoder_input.permute(1,0,2)
                control_input = self.input_dropout_input(context_video).unsqueeze(1)
                control_input = control_input.permute(1,0,2)
                decoder_output, decoder_hidden, _, _ = self.rnn_word(decoder_input, decoder_hidden, control_input)

                decoder_output = decoder_output.permute(1,0,2)
                logprobs = F.log_softmax(
                    self.out(decoder_output.squeeze(1)), dim=1)
                seq_logprobs.append(logprobs.unsqueeze(1))
                hidden_parse_feats_list.append(decoder_output_parse)
                # hidden_feats_list.append(decoder_output)

            seq_logprobs = torch.cat(seq_logprobs, 1)
            # hidden_feats_list = torch.stack(hidden_feats_list,0)
            # hidden_feats_list = hidden_feats_list.squeeze()
            hidden_parse_feats_list = torch.stack(hidden_parse_feats_list,0)
            hidden_parse_feats_list = hidden_parse_feats_list.squeeze()

        elif mode == 'inference':

            for t in range(self.max_length - 1):
                prev_state_parse = list(decoder_hidden_parse)
                prev_state = list(decoder_hidden)
                if len(prev_state_parse)==1:
                    attention_hidden_parse, attention_cell_parse = prev_state_parse[0]
                    attention_hidden, attention_cell = prev_state[0]
                else:
                    for jj in range(len(prev_state_parse)):
                        hidden_item_parse, cell_item_parse = prev_state_parse[jj]
                        if jj > 0:
                            attention_hidden_parse = torch.cat((attention_hidden_parse,hidden_item_parse),-1)
                            attention_cell_parse = torch.cat((attention_cell_parse,cell_item_parse),-1)
                        else:
                            attention_hidden_parse = hidden_item_parse
                            attention_cell_parse = cell_item_parse
                    for jj in range(len(prev_state)):
                        hidden_item, cell_item = prev_state[jj]
                        if jj > 0:
                            attention_hidden = torch.cat((attention_hidden,hidden_item),-1)
                            attention_cell = torch.cat((attention_cell,cell_item),-1)
                        else:
                            attention_hidden = hidden_item
                            attention_cell = cell_item
                context_parse = self.attention_parse(attention_hidden_parse, encoder_outputs_parse)
                context_video = self.attention_video(attention_hidden, encoder_outputs)

                
                if t == 0:  # input <bos>
                    it = torch.LongTensor([self.sos_id] * batch_size).cuda()
                elif sample_max:
                    sampleLogprobs, it = torch.max(logprobs, 1)
                    # seq_logprobs.append(sampleLogprobs.view(-1, 1))
                    it = it.view(-1).long()
                else:
                    if temperature == 1.0:
                        prob_prev = torch.exp(logprobs)
                    else:
                        prob_prev = torch.exp(torch.div(logprobs, temperature))
                    it = torch.multinomial(prob_prev, 1).cuda()
                    sampleLogprobs = logprobs.gather(1, it)
                    seq_logprobs.append(sampleLogprobs.view(-1, 1))
                    it = it.view(-1).long()

                seq_preds.append(it.view(-1, 1))
                xt = self.embedding(it)
                decoder_input_parse = torch.cat([xt,context_parse], dim=1)
                decoder_input_parse = self.input_dropout_input(decoder_input_parse).unsqueeze(1)
                decoder_input_parse = decoder_input_parse.permute(1,0,2)
                control_input_parse = self.input_dropout_input(context_parse).unsqueeze(1)
                control_input_parse = control_input_parse.permute(1,0,2)
                decoder_output_parse, decoder_hidden_parse, _, _ = self.rnn_parse(decoder_input_parse, decoder_hidden_parse, control_input_parse)
                decoder_output_parse = torch.squeeze(decoder_output_parse)

                decoder_input = torch.cat([decoder_output_parse,context_video], dim=1)
                decoder_input = self.input_dropout_input(decoder_input).unsqueeze(1)
                decoder_input = decoder_input.permute(1,0,2)
                control_input = self.input_dropout_input(context_video).unsqueeze(1)
                control_input = control_input.permute(1,0,2)
                decoder_output, decoder_hidden, _, _ = self.rnn_word(decoder_input, decoder_hidden, control_input)
                decoder_output = decoder_output.permute(1,0,2)
                    
                logprobs = F.log_softmax(
                    self.out(decoder_output.squeeze(1)), dim=1)
                seq_logprobs.append(logprobs.unsqueeze(1))
                hidden_parse_feats_list.append(decoder_output_parse)
                # hidden_feats_list.append(decoder_output)

            seq_logprobs = torch.cat(seq_logprobs, 1)
            if template == 0:
                seq_preds = torch.cat(seq_preds[1:], 1)
            else:
                seq_preds = torch.cat(seq_preds, 1)
            # hidden_feats_list = torch.stack(hidden_feats_list,0)
            # hidden_feats_list = hidden_feats_list.squeeze()
            hidden_parse_feats_list = torch.stack(hidden_parse_feats_list,0)
            hidden_parse_feats_list = hidden_parse_feats_list.squeeze()


        return seq_logprobs, seq_preds, hidden_parse_feats_list




    def _init_weights(self):
        """ init the weight of some layers
        """
        nn.init.xavier_normal_(self.out.weight)



    def _cat_directions(self, h):
        """ If the encoder is bidirectional, do the following transformation.
            (#directions * #layers, #batch, dim_hidden) -> (#layers, #batch, #directions * dim_hidden)
        """
        if self.bidirectional_encoder:
            h = torch.cat([h[0:h.size(0):2], h[1:h.size(0):2]], 2)
        return h



    def init_parse_hidden(self, bsz):
        return self.rnn_parse.init_hidden(bsz)



    def init_word_hidden(self, bsz):
        return self.rnn_word.init_hidden(bsz)
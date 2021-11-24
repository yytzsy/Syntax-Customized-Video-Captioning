import torch.nn as nn
import torch
import numpy as np


class EncoderRNN(nn.Module):
    def __init__(self, dim_character, character_vob_size, 
                 dim_latent_code, vocab_size, dim_vid, dim_hidden, input_dropout_p=0.2, rnn_dropout_p=0.5,
                 n_layers=1, bidirectional=False, rnn_cell='gru', 
                 cluster_num = 10, pretrained_extra_cluster_emb = None, pretrained_extra_gs_param = None, pre_trained_tau = 0.9, 
                 word_embedding_weight = None):
        """

        Args:
            hidden_dim (int): dim of hidden state of rnn
            input_dropout_p (int): dropout probability for the input sequence
            dropout_p (float): dropout probability for the output sequence
            n_layers (int): number of rnn layers
            rnn_cell (str): type of RNN cell ('LSTM'/'GRU')
        """
        super(EncoderRNN, self).__init__()
        self.dim_character = dim_character
        self.character_vob_size = character_vob_size
        self.dim_latent_code = dim_latent_code
        self.dim_output = vocab_size
        self.dim_vid = dim_vid
        self.dim_hidden = dim_hidden
        self.input_dropout_p = input_dropout_p
        self.rnn_dropout_p = rnn_dropout_p
        self.n_layers = n_layers
        self.bidirectional = bidirectional
        self.rnn_cell = rnn_cell

        if rnn_cell.lower() == 'lstm':
            self.rnn_cell = nn.LSTM
        elif rnn_cell.lower() == 'gru':
            self.rnn_cell = nn.GRU

        self.vid2hid = nn.Linear(dim_vid, dim_hidden)
        self.temp2hid = nn.Linear(dim_hidden, dim_hidden)
        self.character2hid  = nn.Linear(dim_character,dim_hidden)
        self.sen2hid = nn.Linear(300,dim_hidden)

        self.input_dropout_sentence = nn.Dropout(input_dropout_p)
        self.input_dropout_video = nn.Dropout(input_dropout_p)
        self.input_dropout_temp = nn.Dropout(input_dropout_p)
        self.input_dropout_character = nn.Dropout(input_dropout_p)

        self.character_embedding = nn.Embedding(character_vob_size, dim_character)
        self.rnn_character = self.rnn_cell(input_size = dim_hidden, hidden_size = dim_hidden/2, num_layers = n_layers, batch_first=True,
                                bidirectional=True, dropout=self.rnn_dropout_p)

        self.word_semantic_embedding = nn.Embedding(vocab_size, 300)
        if word_embedding_weight is not None:
            self.word_semantic_embedding.weight = nn.Parameter(torch.FloatTensor(word_embedding_weight))
            self.word_semantic_embedding.weight.requires_grad = False

        self.rnn_sentence = self.rnn_cell(input_size = dim_hidden, hidden_size = dim_hidden, num_layers = n_layers, batch_first=True,
                                bidirectional=bool(bidirectional), dropout=self.rnn_dropout_p)


        self.rnn_video = self.rnn_cell(input_size = dim_hidden, hidden_size = dim_hidden, num_layers = n_layers, batch_first=True,
                                bidirectional=bool(bidirectional), dropout=self.rnn_dropout_p)


        self.rnn_syntax = self.rnn_cell(input_size = dim_hidden, hidden_size = dim_hidden, num_layers = n_layers, batch_first=True,
                                bidirectional=bool(bidirectional), dropout=self.rnn_dropout_p)

        self._init_hidden()



    def _init_hidden(self):
        nn.init.xavier_normal_(self.vid2hid.weight)
        nn.init.xavier_normal_(self.temp2hid.weight)
        nn.init.xavier_normal_(self.character2hid.weight)
        nn.init.xavier_normal_(self.sen2hid.weight)



    def forward(self, vid_feats, labels_character, labels_character_random, labels_character_template, labels_sentence, labels_template_sentence):

        if vid_feats is not None:
            batch_size, seq_len, dim_vid = vid_feats.size()
            vid_feats = self.vid2hid(vid_feats.view(-1, dim_vid))
            vid_feats = self.input_dropout_video(vid_feats)
            vid_feats = vid_feats.view(batch_size, seq_len, self.dim_hidden)
            self.rnn_video.flatten_parameters()
            video_output, video_hidden = self.rnn_video(vid_feats)
        else:
            video_output = None

        # encode input sentence
        if labels_sentence is not None:
            sentence_feats = self.word_semantic_embedding(labels_sentence)
            batch_size, word_num, dim_word = sentence_feats.size()
            sentence_feats = sentence_feats.contiguous().view(-1, dim_word)
            sentence_feats = self.sen2hid(sentence_feats)
            _,dim_hidden = sentence_feats.size()
            sentence_feats = sentence_feats.contiguous().view(batch_size, word_num, dim_hidden)
            sentence_output = sentence_feats
        else:
            sentence_output = None
            sentence_hidden = None
            
        # encode input sentence
        if labels_template_sentence is not None:
            template_sentence_feats = self.word_semantic_embedding(labels_template_sentence)
            batch_size, word_num, dim_word = template_sentence_feats.size()
            template_sentence_feats = template_sentence_feats.contiguous().view(-1, dim_word)
            template_sentence_feats = self.sen2hid(template_sentence_feats)
            _,dim_hidden = template_sentence_feats.size()
            template_sentence_feats = template_sentence_feats.contiguous().view(batch_size, word_num, dim_hidden)
            template_sentence_output = template_sentence_feats
        else:
            template_sentence_output = None
            template_sentence_hidden = None            


        # encode origin syntax
        if labels_character is not None:
            labels_character_input = labels_character.clone()
            random_list = np.random.rand(batch_size,)
            for i in range(batch_size):
                if random_list[i] >= 0.3:
                    labels_character_input[i] = labels_character_random[i].clone()

            batch_size, word_len, character_len = labels_character_input.size()
            reshape_labels_character = labels_character_input.view(batch_size*word_len,character_len)
            character_feats = self.character_embedding(reshape_labels_character)  #batch_size*word_len,character_len,dim_character
            character_feats = self.character2hid(character_feats.view(-1,self.dim_character)) #batch_size*word_len*character_len,dim_hidden
            character_feats = self.input_dropout_character(character_feats)
            character_feats = character_feats.view(batch_size*word_len, character_len, self.dim_hidden)
            self.rnn_character.flatten_parameters()
            character_output, character_hidden = self.rnn_character(character_feats) #batch_size*word_len, character_len, dim_hidden(including back and forward)
            word_feats = torch.mean(character_output,1)
            word_feats = word_feats.view(batch_size,word_len,self.dim_hidden)

            syntax_feats = self.temp2hid(word_feats.view(-1, self.dim_hidden))
            syntax_feats = self.input_dropout_temp(syntax_feats)
            syntax_feats = syntax_feats.view(batch_size, word_len, self.dim_hidden)
            self.rnn_syntax.flatten_parameters()
            syntax_output, syntax_hidden = self.rnn_syntax(syntax_feats)
        else:
            syntax_output = None
            syntax_hidden = None

        # encode template syntax
        if labels_character_template is not None:
            reshape_labels_character_template = labels_character_template.view(batch_size*word_len,character_len)
            character_feats_template = self.character_embedding(reshape_labels_character_template)  #batch_size*word_len,character_len,dim_character
            character_feats_template = self.character2hid(character_feats_template.view(-1,self.dim_character)) #batch_size*word_len*character_len,dim_hidden
            character_feats_template = self.input_dropout_character(character_feats_template)
            character_feats_template = character_feats_template.view(batch_size*word_len, character_len, self.dim_hidden)
            character_output_template, character_hidden_template = self.rnn_character(character_feats_template) #batch_size*word_len, character_len, dim_hidden(including back and forward)
            word_feats_template = torch.mean(character_output_template,1)
            word_feats_template = word_feats_template.view(batch_size,word_len,self.dim_hidden)

            template_syntax_feats = self.temp2hid(word_feats_template.view(-1, self.dim_hidden))
            template_syntax_feats = self.input_dropout_temp(template_syntax_feats)
            template_syntax_feats = template_syntax_feats.view(batch_size, word_len, self.dim_hidden)
            template_syntax_output, template_syntax_hidden = self.rnn_syntax(template_syntax_feats)
        else:
            template_syntax_output = None
            template_syntax_hidden = None


        return video_output, sentence_output, template_sentence_output, syntax_output, template_syntax_output
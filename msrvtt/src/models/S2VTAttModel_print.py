import torch.nn as nn
import torch
import numpy as np

class S2VTAttModel_print(nn.Module):
    def __init__(self, encoder, decoder, decoder_sentence, decoder_syntax, dim_hidden):
        """

        Args:
            encoder (nn.Module): Encoder rnn
            decoder (nn.Module): Decoder rnn
        """
        super(S2VTAttModel_print, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.decoder_sentence = decoder_sentence
        self.decoder_syntax = decoder_syntax
        self.dim_hidden = dim_hidden

        self.vid2hid = nn.Linear(dim_hidden, dim_hidden)
        nn.init.xavier_normal_(self.vid2hid.weight)

    def forward(self, vid_feats, target_variable, target_variable_tepmlate, labels_character, labels_random_character, labels_character_template, target_variable_parse, target_variable_parse_template, mode='train', opt={}):
        """

        Args:
            vid_feats (Variable): video feats of shape [batch_size, seq_len, dim_vid]
            target_variable (None, optional): groung truth labels

        Returns:
            seq_prob: Variable of shape [batch_size, max_len-1, vocab_size]
            seq_preds: [] or Variable of shape [batch_size, max_len-1]
        """
        #encode
        
        video_outputs, sentence_outputs, template_sentence_outputs, syntax_encoder_outputs, template_syntax_encoder_outputs = \
        self.encoder(vid_feats,labels_character,labels_random_character,labels_character_template,target_variable,target_variable_tepmlate)
        batch_size,_,_ = np.shape(video_outputs)

        mean_video_outputs = torch.mean(video_outputs,1)
        mean_sentence_outputs = torch.mean(sentence_outputs,1)
        mean_template_sentence_outputs = torch.mean(template_sentence_outputs,1)
        return mean_video_outputs, mean_sentence_outputs, mean_template_sentence_outputs

import torch.nn as nn
import torch
import numpy as np

class S2VTAttModel(nn.Module):
    def __init__(self, encoder, decoder, decoder_sentence, decoder_syntax, dim_hidden):
        """

        Args:
            encoder (nn.Module): Encoder rnn
            decoder (nn.Module): Decoder rnn
        """
        super(S2VTAttModel, self).__init__()
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

        seq_prob, seq_preds, origin_syntax_vec = self.decoder(video_outputs, syntax_encoder_outputs, target_variable, mode, opt)

        if mode == 'train':

            # decode sentence from sentence semantics
            seq_prob_sen, seq_preds_sen, origin_syntax_vec1 = self.decoder(sentence_outputs, syntax_encoder_outputs, target_variable, mode, opt)
            seq_prob_sen_template, seq_preds_sen_template, template_syntax_vec1 = self.decoder(template_sentence_outputs, template_syntax_encoder_outputs, target_variable_tepmlate, mode, opt)

            # syntax supervision
            origin_syntax_vec = origin_syntax_vec.permute(1,0,2)
            seq_prob_syntax, seq_preds_syntax = self.decoder_syntax(origin_syntax_vec, None, target_variable_parse, mode, opt)
            origin_syntax_vec1 = origin_syntax_vec1.permute(1,0,2)
            seq_prob_syntax1, seq_preds_syntax1 = self.decoder_syntax(origin_syntax_vec1, None, target_variable_parse, mode, opt)

            _, _, template_syntax_vec = self.decoder(video_outputs, template_syntax_encoder_outputs, None, 'inference', opt)
            template_syntax_vec = template_syntax_vec.permute(1,0,2)
            seq_prob_syntax_template, seq_preds_syntax_template = self.decoder_syntax(template_syntax_vec, None, target_variable_parse_template, mode, opt)
            template_syntax_vec1 = template_syntax_vec1.permute(1,0,2)
            seq_prob_syntax_template1, seq_preds_syntax_template1 = self.decoder_syntax(template_syntax_vec1, None, target_variable_parse_template, mode, opt)

            return seq_prob, seq_prob_sen, seq_prob_sen_template, seq_prob_syntax, seq_prob_syntax1, seq_prob_syntax_template, seq_prob_syntax_template1
        else:
            return seq_prob, seq_preds

import json
import os

import numpy as np

import misc.utils as utils
import opts
import torch
import torch.optim as optim
from torch.nn.utils import clip_grad_value_
from dataloader import VideoDataset
from misc.rewards import get_self_critical_reward, init_cider_scorer
from models import DecoderRNN, DecoderSentenceRNN, DecoderSyntaxRNN, EncoderRNN, S2VTAttModel, S2VTModel
from torch import nn
from torch.utils.data import DataLoader


import sys
reload(sys)
sys.setdefaultencoding('utf-8')

def train(loader, model, crit, optimizer, lr_scheduler, opt, rl_crit=None, start_epoch = 0):
    model.train()
    #model = nn.DataParallel(model)
    for epoch in range(opt["epochs"]):
        torch.cuda.empty_cache()
        lr_scheduler.step()

        current_epoch = epoch + start_epoch
        if current_epoch >= 50:
            alpha = 1.0
        else:
            alpha = 0.0

        iteration = 0
        # If start self crit training
        if opt["self_crit_after"] != -1 and current_epoch >= opt["self_crit_after"]:
            sc_flag = True
            init_cider_scorer(opt["cached_tokens"])
        else:
            sc_flag = False

        for data in loader:
            torch.cuda.synchronize()
            print os.path.abspath('./').split('/')[-1]

            fc_feats = data['fc_feats'].cuda()
            labels = data['labels'].cuda()
            labels_random_character = data['labels_random_character'].cuda()
            labels_character = data['labels_character'].cuda()  #batchsize, words number in sentences, max_character_len
            masks = data['masks'].cuda()
            labels_parse = data['label_parse'].cuda()
            masks_parse = data['mask_parse'].cuda()
            
            labels_template = data['train_template_labels'].cuda()
            masks_template = data['train_mask_template'].cuda()
            labels_template_character = data['train_template_labels_character'].cuda()
            labels_template_parse = data['train_template_parse_labels'].cuda()
            labels_template_parse_mask = data['train_template_parse_masks'].cuda()

            optimizer.zero_grad()
            if not sc_flag:
                seq_prob, seq_prob_sen, seq_prob_sen_template, seq_prob_syntax, seq_prob_syntax1, seq_prob_syntax_template, seq_prob_syntax_template1 = model(fc_feats, labels, labels_template, labels_character, labels_random_character, labels_template_character, labels_parse, labels_template_parse, 'train', opt)
                
                loss_caption = crit(seq_prob, labels[:, 1:], masks[:, 1:])
                loss_paired_sentence = crit(seq_prob_sen, labels[:, 1:], masks[:, 1:])
                loss_template_sentence = crit(seq_prob_sen_template, labels_template[:, 1:], masks_template[:, 1:])

                loss_syntax = crit(seq_prob_syntax, labels_parse[:, 1:], masks_parse[:, 1:]) + crit(seq_prob_syntax1, labels_parse[:, 1:], masks_parse[:, 1:])
                loss_syntax_template = alpha * crit(seq_prob_syntax_template, labels_template_parse[:, 1:], labels_template_parse_mask[:, 1:]) + crit(seq_prob_syntax_template1, labels_template_parse[:, 1:], labels_template_parse_mask[:, 1:]) 

                loss = loss_caption + loss_paired_sentence + loss_template_sentence + loss_syntax + loss_syntax_template

            loss.backward()
            clip_grad_value_(model.parameters(), opt['grad_clip'])
            optimizer.step()
            train_loss = loss.item()
            torch.cuda.synchronize()
            iteration += 1

            if not sc_flag:
                print("iter %d (current_epoch %d), loss_caption = %.2f, loss_paired_sentence = %.2f, loss_template_sentence = %.2f, loss_syntax = %.2f, loss_syntax_template = %.2f, train_loss = %.2f" % (iteration, current_epoch, loss_caption, loss_paired_sentence, loss_template_sentence, loss_syntax, loss_syntax_template, train_loss))
            else:
                print("iter %d (current_epoch %d), avg_reward = %.2f" %
                      (iteration, current_epoch, np.mean(reward[:, 0])))

        if current_epoch % opt["save_checkpoint_every"] == 0:
            model_path = os.path.join(opt["checkpoint_path"],
                                      'model_%d.pth' % (current_epoch))
            model_info_path = os.path.join(opt["checkpoint_path"],
                                           'model_score.txt')
            torch.save(model.state_dict(), model_path)
            print("model saved to %s" % (model_path))
            with open(model_info_path, 'a') as f:
                f.write("(current_epoch %d), loss_caption = %.2f, loss_paired_sentence = %.2f,  loss_template_sentence = %.2f, loss_syntax = %.2f, loss_syntax_template = %.2f, train_loss = %.2f \n" % (current_epoch, loss_caption, loss_paired_sentence, loss_template_sentence, loss_syntax, loss_syntax_template, train_loss))


def main(opt):
    dataset = VideoDataset(opt, 'train')
    dataloader = DataLoader(dataset, batch_size=opt["batch_size"], shuffle=True)
    opt["vocab_size"] = dataset.get_vocab_size()
    opt["parse_size"] = dataset.get_parse_vocab_size()
    opt["character_size"] = dataset.get_character_size()

    glove_word_embedding = np.load(opt['glove_word_emb'])
    glove_word_embedding = glove_word_embedding.tolist()

    if opt["model"] == 'S2VTModel':
        model = S2VTModel(
            opt["vocab_size"],
            opt["max_len"],
            opt["dim_hidden"],
            opt["dim_word"],
            opt['dim_vid'],
            rnn_cell=opt['rnn_type'],
            n_layers=opt['num_layers'],
            rnn_dropout_p=opt["rnn_dropout_p"])
    elif opt["model"] == "S2VTAttModel":
        encoder = EncoderRNN(
            opt["dim_character"],
            opt["character_size"],
            opt["dim_latent_code"],
            opt["vocab_size"],
            opt["dim_vid"],
            opt["dim_hidden"],
            input_dropout_p=opt["input_dropout_p"],
            rnn_dropout_p=opt["rnn_dropout_p"],
            n_layers=opt["num_layers"],
            bidirectional=opt["bidirectional"],
            rnn_cell=opt['rnn_type'],
            cluster_num = opt["cluster_num"],
            pretrained_extra_cluster_emb = None,
            pretrained_extra_gs_param = None,
            pre_trained_tau = np.array(opt['tau']),
            word_embedding_weight = glove_word_embedding
            )
        decoder = DecoderRNN(
            opt["vocab_size"],
            opt["max_len"],
            opt["dim_hidden"],
            opt["dim_word"],
            opt["dim_parse"],
            n_layers=opt["num_layers"],
            input_dropout_p=opt["input_dropout_p"],
            rnn_dropout_p=opt["rnn_dropout_p"],
            bidirectional=opt["bidirectional"],
            w_dropout_p=opt["w_dropout_p"],
            chunk_size=opt["chunk_size"],
            embedding_pretrained_weights = None
            )
        decoder_sentence = DecoderSentenceRNN(
            opt["vocab_size"],
            opt["max_len"],
            opt["dim_hidden"],
            opt["dim_word"],
            opt["dim_parse"],
            n_layers=opt["num_layers"],
            input_dropout_p=opt["input_dropout_p"],
            rnn_cell=opt['decode_rnn_type'],
            rnn_dropout_p=opt["rnn_dropout_p"],
            bidirectional=opt["bidirectional"],
            w_dropout_p=opt["w_dropout_p"],
            chunk_size=opt["chunk_size"])
        decoder_syntax = DecoderSyntaxRNN(
            opt['parse_size'],
            opt['parse_max_len'],
            opt['dim_hidden'],
            opt['dim_parse'],
            n_layers=opt["num_layers"],
            rnn_cell=opt['decode_rnn_type'],
            bidirectional=opt["bidirectional"],
            input_dropout_p=opt["input_dropout_p"],
            rnn_dropout_p=opt["rnn_dropout_p"],
            )
        model = S2VTAttModel(encoder, decoder, decoder_sentence, decoder_syntax, opt['dim_hidden'])
        print model
    model = model.cuda()
    crit = utils.LanguageModelCriterion()
    rl_crit = utils.RewardCriterion()
    optimizer = optim.Adam(
        model.parameters(),
        lr=opt["learning_rate"],
        weight_decay=opt["weight_decay"])
    exp_lr_scheduler = optim.lr_scheduler.StepLR(
        optimizer,
        step_size=opt["learning_rate_decay_every"],
        gamma=opt["learning_rate_decay_rate"])

    start_epoch = 0
    # model_path = './save/model_'+str(start_epoch-1)+'.pth'
    # model.load_state_dict(torch.load(model_path))
    train(dataloader, model, crit, optimizer, exp_lr_scheduler, opt, rl_crit, start_epoch)


if __name__ == '__main__':
    opt = opts.parse_opt()
    opt = vars(opt)
    os.environ['CUDA_VISIBLE_DEVICES'] = opt["gpu"]
    opt_json = os.path.join(opt["checkpoint_path"], 'opt_info.json')
    if not os.path.isdir(opt["checkpoint_path"]):
        os.mkdir(opt["checkpoint_path"])
    with open(opt_json, 'w') as f:
        json.dump(opt, f)
    print('save opt details to %s' % (opt_json))
    main(opt)

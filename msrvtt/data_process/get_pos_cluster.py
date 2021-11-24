import json
import random
import os
import numpy as np
import h5py


info_extend = json.load(open('../msrvtt_data/msrvtt_coco_info_extend.json'))
ix_to_word = info_extend['ix_to_word']
word_to_ix = info_extend['word_to_ix']

word_pos_dict = {}
pos_word_dict = {}
all_captions = json.load(open('../msrvtt_data/msrvtt_caption_with_template.json'))
count=0
for video in all_captions:
	count+=1
	print count
	captions_list = all_captions[video]['final_captions']
	template_captions_list = all_captions[video]['template_final_captions']
	pos_list = all_captions[video]['final_pos']
	template_pos_list = all_captions[video]['template_pos']
	for idx,caption in enumerate(captions_list):
		pos = pos_list[idx]
		for jj in range(len(caption)):
			word = caption[jj]
			if word == '<sos>' or word == '<eos>':
				continue
			word_pos_dict[word] = pos[jj]
			if pos[jj] not in pos_word_dict:
				pos_word_dict[pos[jj]] = [word]
			else:
				pos_word_dict[pos[jj]].append(word)
	for idx,caption in enumerate(template_captions_list):
		pos = template_pos_list[idx]
		for jj in range(len(caption)):
			word = caption[jj]
			if word == '<sos>' or word == '<eos>':
				continue
			word_pos_dict[word] = pos[jj]
			if pos[jj] not in pos_word_dict:
				pos_word_dict[pos[jj]] = [word]
			else:
				pos_word_dict[pos[jj]].append(word)


for pos in pos_word_dict:
	pos_word_dict[pos] = set(pos_word_dict[pos])
np.save('word_pos_dict.npy',word_pos_dict)
np.save('pos_word_dict.npy',pos_word_dict)

pos_idx_dict = {}
for pos in pos_word_dict:
	content = []
	for word in pos_word_dict[pos]:
		content.append(word_to_ix[word])
	pos_idx_dict[pos] = content
np.save('pos_idx_dict.npy',pos_idx_dict)
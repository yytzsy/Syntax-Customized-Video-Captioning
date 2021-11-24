import json
import random
import os
import numpy as np
import h5py
from nltk.corpus import stopwords 

stop_words = set(stopwords.words('english'))
glove_word_embed_path = '../msrvtt_data/glove.840B.300d_dict.npy'
glove_word_emb = np.load(glove_word_embed_path)
glove_word_emb = glove_word_emb.tolist()

info_extend = json.load(open('../msrvtt_data/msrvtt_coco_info_extend.json'))
ix_to_word = info_extend['ix_to_word']

ix = 0
ix_to_emb = {}
word_to_emb = {}
for item in ix_to_word:
	current_word = ix_to_word[item]
	if current_word in glove_word_emb:
		ix_to_emb[item] = glove_word_emb[current_word]
		word_to_emb[current_word] = glove_word_emb[current_word]
		ix+=1
		print ix
	else:
		ix_to_emb[item] = np.zeros(300,)+0.0
		word_to_emb[current_word] = np.zeros(300,)+0.0

print len(ix_to_emb)
print len(word_to_emb)
np.save(open('./ix_to_emb.npy','w'),ix_to_emb)
np.save(open('./word_to_emb.npy','w'),word_to_emb)


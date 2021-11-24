import json
import numpy as np



glove_dict = np.load('../msrvtt_data/glove.840B.300d_dict.npy')
glove_dict = glove_dict.tolist()
print 'glove loaded!'

info_extend = json.load(open('../msrvtt_data/msrvtt_coco_info_extend.json'))
ix_to_word = info_extend['ix_to_word']

word_embedding = np.zeros((len(ix_to_word),300))+0.0
for ix in range(len(ix_to_word)):
    word = ix_to_word[str(ix)]
    if word in glove_dict:
        word_embedding[ix] = glove_dict[word]
        print word


np.save(open('word_embedding_from_glove.npy','w'),word_embedding)

import json
import random
import os
import numpy as np
import h5py
from nltk.corpus import stopwords 


cha2ix = {}
cha2ix['<eow>'] = 0
cha2ix['<sow>'] = 1
cha2ix['a'] = 2
cha2ix['b'] = 3
cha2ix['c'] = 4
cha2ix['d'] = 5
cha2ix['e'] = 6
cha2ix['f'] = 7
cha2ix['g'] = 8
cha2ix['h'] = 9
cha2ix['i'] = 10
cha2ix['j'] = 11
cha2ix['k'] = 12
cha2ix['l'] = 13
cha2ix['m'] = 14
cha2ix['n'] = 15
cha2ix['o'] = 16
cha2ix['p'] = 17
cha2ix['q'] = 18
cha2ix['r'] = 19
cha2ix['s'] = 20
cha2ix['t'] = 21
cha2ix['u'] = 22
cha2ix['v'] = 23
cha2ix['w'] = 24
cha2ix['x'] = 25
cha2ix['y'] = 26
cha2ix['z'] = 27
cha2ix['<'] = 28
cha2ix['>'] = 29
cha2ix['UNK'] = 30

np.save(open('cha2ix.npy','w'),cha2ix)

ix2cha = {}
ix2cha[0] = '<eow>'
ix2cha[1] = '<sow>'
ix2cha[2] = 'a'
ix2cha[3] = 'b'
ix2cha[4] = 'c'
ix2cha[5] = 'd'
ix2cha[6] = 'e'
ix2cha[7] = 'f'
ix2cha[8] = 'g'
ix2cha[9] = 'h'
ix2cha[10] = 'i'
ix2cha[11] = 'j'
ix2cha[12] = 'k'
ix2cha[13] = 'l'
ix2cha[14] = 'm'
ix2cha[15] = 'n'
ix2cha[16] = 'o'
ix2cha[17] = 'p'
ix2cha[18] = 'q'
ix2cha[19] = 'r'
ix2cha[20] = 's'
ix2cha[21] = 't'
ix2cha[22] = 'u'
ix2cha[23] = 'v'
ix2cha[24] = 'w'
ix2cha[25] = 'x'
ix2cha[26] = 'y'
ix2cha[27] = 'z'
cha2ix[28] = '<'
cha2ix[29] = '>'
cha2ix[30] = 'UNK'


np.save(open('ix2cha.npy','w'),ix2cha)
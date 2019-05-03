#!/bin/python

from __future__ import print_function

import common

########################################
print('loading vocab files')
import pickle
vocabs={}
total_words={}
for ccTLD in common.ccTLDs:
    print('  '+ccTLD)
    vocab_filename='crawls/ccTLD.'+ccTLD+'.jl.gz.dedupe.vocab'
    with open(vocab_filename,'r') as f:
        vocabs[ccTLD]=pickle.load(f)
        total_words[ccTLD]=sum(vocabs[ccTLD].values())
        break

########################################
print('printing stats')

with open('words.txt','r') as f:
    for word in f:
        word=word.strip().decode('utf-8').encode('unicode_escape').decode('unicode_escape')
        print(word.encode('utf-8'))
        for ccTLD in sorted(vocabs.keys()):
            print('  %s : %10d / %10d = %0.2E'%(
                ccTLD,
                vocabs[ccTLD][word],
                total_words[ccTLD],
                vocabs[ccTLD][word]/float(total_words[ccTLD]),
                ))
        print()


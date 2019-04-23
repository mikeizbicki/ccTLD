#!/bin/python

from __future__ import print_function

# command line args
import argparse
parser=argparse.ArgumentParser('extract stats from file containing containers.Counter object')
parser.add_argument('--filename',type=str,required=True)
parser.add_argument('--dictsize',type=int,default=2**20)
parser.add_argument('--outputdict',type=str,default=None)
args = parser.parse_args()

# print stats
print('analyzing',args.filename)
import pickle

with open(args.filename,'r') as f:
    vocab=pickle.load(f)

    # print basic stats
    total_vocab=len(vocab.keys())
    total_words=sum(vocab.values())
    print('total_vocab=',total_vocab)
    print('total_words=',total_words)

    # print most popular words
    print('most popular:')
    popular=vocab.most_common(100)
    for (word,count) in popular:
        if len(word)<5:
            continue
        print('  %20s : %10d / %d = %0.4E'%(
            word.encode('utf-8'),
            count,
            total_words,
            float(count)/total_words,
            ))

    # output dictionary
    if args.outputdict is not None:
        with open(args.outputdict,'w') as f:
            output=dict(zip(map(lambda (x,y):x, vocab.most_common(args.dictsize)),range(args.dictsize)))
            pickle.dump(output,f)

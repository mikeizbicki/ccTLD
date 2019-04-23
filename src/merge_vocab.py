#!/bin/python

from __future__ import print_function

# command line args
import argparse
parser=argparse.ArgumentParser('extract vocab from .jl file')
parser.add_argument('--filenames',type=str,nargs='*',required=True)
parser.add_argument('--output',type=str,required=True)
args = parser.parse_args()

########################################
print('loading vocabs')

import pickle
vocabs=[]
for filename in args.filenames:
    print('  ',filename)
    with open(filename,'r') as f:
        vocabs.append(pickle.load(f))

########################################
print('outputing vocab')

from functools import reduce
sum_vocabs=reduce(lambda x,y:x+y,vocabs)
with open(args.output,'w') as f:
    pickle.dump(sum_vocabs,f)

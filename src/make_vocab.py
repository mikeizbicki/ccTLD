#!/bin/python

from __future__ import print_function

# command line args
import argparse
parser=argparse.ArgumentParser('extract vocab from .jl file')
parser.add_argument('--filename',type=str,required=True)
args = parser.parse_args()

# loop through file
import simplejson as json
from collections import Counter
import re
import datetime
import pickle
from urlparse import urlparse

vocab=Counter()
domains=Counter()
with open(args.filename,'r') as f:
    count=0
    for line in f:
        count+=1
        if count%10000==0:
            print(datetime.datetime.now(),'count=',count)

        page=json.loads(line)
        text=page['text']
        text=text.lower()
        #print('text=',text)
        #tokens=re.findall(r"\w+|[^\w]", text, re.UNICODE)
        tokens=re.findall(r"\w+|[^\w\s]", text, re.UNICODE)
        #print('tokens=',tokens)
        vocab.update(tokens)
        #print('vocab=',vocab)

        domains.update([urlparse(page['url']).hostname])
        #print('domains=',domains)

# save output
filename_vocab=args.filename+'.vocab'
with open(filename_vocab,'w') as f:
    print('saving output to '+filename_vocab)
    pickle.dump(vocab,f)

filename_domain=args.filename+'.domain'
with open(filename_domain,'w') as f:
    print('saving output to '+filename_domain)
    pickle.dump(domains,f)

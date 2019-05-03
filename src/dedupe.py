#!/bin/python

from __future__ import print_function

# command line args
import argparse
parser=argparse.ArgumentParser('remove duplicate entries from .jl.gz file')
parser.add_argument('--filename',type=str,required=True)
args = parser.parse_args()

# loop through file
import simplejson as json
import gzip
import datetime
import psutil
import os
process = psutil.Process(os.getpid())

output_filename=args.filename+'.dedupe'
if os.path.isfile('output_filename'):
    print('aborting, file exists: ',output_file)
    import sys
    sys.exit(1)

linecount=0
dupes=0
texts=set()
with gzip.open(args.filename,'r') as infile:
    with gzip.open(output_filename,'w') as outfile:
        for line in infile:
            data=json.loads(line)

            text_hash=hash(data['text'])
            if text_hash not in texts:
                outfile.write(line)
                texts.add(text_hash)
            else:
                dupes+=1

            # print status info
            if linecount%10000==0:
                print("%s: linecount=%i  dupe_frac=%0.2f  mem=%0.2fMB"%(
                    str(datetime.datetime.now()),
                    linecount,
                    float(dupes)/(linecount+1),
                    process.memory_info().rss/1e6,
                    #process.get_memory_info()[0]/1e6,
                    ))
            linecount+=1

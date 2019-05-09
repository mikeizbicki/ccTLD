#!/bin/python

from __future__ import print_function

########################################
import argparse
parser=argparse.ArgumentParser('train a skipgram model')
parser.add_argument('--batch_size',type=int,default=128)
parser.add_argument('--shuffle_size_1',type=int,default=2**20)
parser.add_argument('--shuffle_size_2',type=int,default=2**10)
parser.add_argument('--shuffle_size_3',type=int,default=2**14)
parser.add_argument('--shuffle_fast',action='store_true')
parser.add_argument('--vocab_size',type=int,default=2**20)
parser.add_argument('--embedding_size',type=int,default=300)
parser.add_argument('--num_embeddings',type=int,default=5)
parser.add_argument('--nce_samples',type=int,default=2**8)
parser.add_argument('--context_left',type=int,default=3)
parser.add_argument('--context_right',type=int,default=3)
parser.add_argument('--learning_rate',type=float,default=5e-4)
parser.add_argument('--reg',type=float,default=0.0)
parser.add_argument('--output_dir',type=str,default=None)
parser.add_argument('--data_source',type=str,choices=['crawls','billion'],default='crawls')
args = parser.parse_args()

if args.shuffle_fast:
    args.shuffle_size_1=10
    args.shuffle_size_2=10
    args.shuffle_size_3=10

if args.output_dir is not None:
    import os
    try:
        os.makedirs(args.output_dir)
    except:
        pass
    import simplejson as json
    args_str=json.dumps(vars(args))
    with open(args.output_dir+'/args.json','w') as f:
        f.write(args_str)

########################################
import tensorflow as tf
import skipgram

tf.logging.set_verbosity(tf.logging.INFO)
estimator=tf.estimator.Estimator(
    model_fn=skipgram.model_fn_train,
    model_dir=args.output_dir,
    params=args,
    )

estimator.train(
    input_fn=lambda: skipgram.input_fn_train(args)
    )

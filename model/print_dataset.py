#!/bin/python

from __future__ import print_function

########################################
import argparse
parser=argparse.ArgumentParser('train a skipgram model')
parser.add_argument('--batch_size',type=int,default=32)
parser.add_argument('--shuffle_size',type=int,default=2**14)
parser.add_argument('--vocab_size',type=int,default=2**20)
parser.add_argument('--embedding_size',type=int,default=300)
parser.add_argument('--num_embeddings',type=int,default=5)
parser.add_argument('--nce_samples',type=int,default=2**14)
parser.add_argument('--context_left',type=int,default=3)
parser.add_argument('--context_right',type=int,default=3)
parser.add_argument('--reg_l1',type=float,default=0.0)
parser.add_argument('--reg_l2',type=float,default=0.0)
parser.add_argument('--output_dir',type=str,default=None)
args = parser.parse_args()

########################################
print('loading tensorflow')
import tensorflow as tf
import skipgram

dataset=skipgram.input_fn_train(args)
iterator = dataset.make_initializable_iterator()
next_op=iterator.get_next()

print('starting session')
with tf.Session() as sess:
    sess.run(iterator.initializer)
    sess.run(tf.initialize_all_tables())

    for i in range(20):
        next=sess.run(next_op)
        print('  next=',next)

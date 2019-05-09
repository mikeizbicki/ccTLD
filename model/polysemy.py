#!/bin/python

'''
This file prints the words with the most divergent embeddings.
'''

from __future__ import print_function

########################################
import argparse
parser=argparse.ArgumentParser('plot the cluster of ccTLDs induced by a model')
parser.add_argument('--model_dir',type=str,required=True)
parser.add_argument('--output',type=str,default='cluster')
args = parser.parse_args()

with open(args.model_dir+'/args.json','r') as f:
    import simplejson as json
    data=json.loads(f.readline())
    args_train=type('',(),data)

########################################
print('create vocab')
import common
vocab_top=common.get_vocab(args_train.vocab_size)

########################################
print('loading tensorflow')
import tensorflow as tf
import numpy as np
import common
num_ccTLD=len(common.ccTLDs)

embeddings = tf.get_variable(
    name='embeddings',
    shape=[args_train.vocab_size,args_train.embedding_size,args_train.num_embeddings],
    )
embeddings_projector = tf.get_variable(
    name='embeddings_projector',
    shape=[args_train.num_embeddings,num_ccTLD],
    )

with tf.variable_scope('word_counter'):
    counter_words_ccTLD=tf.get_variable(
        name='counter_words_ccTLD',
        initializer=tf.zeros([args_train.vocab_size,len(common.ccTLDs)],dtype=tf.int32),
        #initializer=tf.zeros([args.vocab_size],dtype=tf.int32),
        dtype=tf.int32,
        trainable=False,
        #collections=[tf.GraphKeys.LOCAL_VARIABLES],
        use_resource=True,
        )
    counter_words=tf.reduce_sum(counter_words_ccTLD,axis=1)

embeddings_mean = tf.reduce_mean(embeddings,axis=2)
embeddings_mean_reshape= tf.reshape(embeddings_mean,[args_train.vocab_size,args_train.embedding_size,1])
embeddings_diff = embeddings_mean_reshape - embeddings

embeddings_diff_cos = tf.reduce_sum(
    tf.nn.l2_normalize(embeddings_mean_reshape,axis=1)*tf.nn.l2_normalize(embeddings_diff,axis=1),
    axis=1,
    )

embeddings_diff_l2 = tf.reduce_sum(embeddings_diff*embeddings_diff,axis=1)

print('embeddings_diff_l2=',embeddings_diff_l2)

embeddings_l1 = tf.reduce_sum(tf.abs(embeddings_diff_cos),axis=1)

saver = tf.train.Saver([embeddings,embeddings_projector,counter_words_ccTLD])

########################################
print('calculating polysemy')

with tf.Session() as sess:
    chkpt_file=tf.train.latest_checkpoint(args.model_dir)
    #from tensorflow.python.tools.inspect_checkpoint import print_tensors_in_checkpoint_file
    #print_tensors_in_checkpoint_file(chkpt_file, all_tensors=True, tensor_name='')
    saver.restore(sess, chkpt_file)

    scores,counts=sess.run([embeddings_l1,counter_words])

highest_scores=np.argsort(scores)
#print('highest_scores=',highest_scores[:100])

total_printed=0
for i in range(args_train.vocab_size):
    #index=highest_scores[args_train.vocab_size-i-1]
    index=highest_scores[i]
    if counts[index]>1000 and len(vocab_top[index])>5:
        print('  %12s : %0.2f : %5d'%(vocab_top[index].encode('utf-8'),scores[index],counts[index]))
        total_printed+=1

    if total_printed>100:
        break

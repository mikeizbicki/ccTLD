#!/bin/python

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

saver = tf.train.Saver([embeddings,embeddings_projector])

with tf.Session() as sess:
    chkpt_file=tf.train.latest_checkpoint(args.model_dir)
    saver.restore(sess, chkpt_file)
    ccTLD_embedding=np.transpose(sess.run(embeddings_projector))

########################################
print('making plot')
from scipy.cluster.hierarchy import dendrogram, linkage
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt

plt.figure(figsize=(10, 7))
dendrogram(
    linkage(ccTLD_embedding,'ward',optimal_ordering=True),
    orientation='top',
    labels=common.ccTLDs,
    distance_sort='descending',
    show_leaf_counts=True
    )
plt.savefig(args.output+'_dendrogram.png')

########################################
from sklearn.decomposition import PCA

pca = PCA(n_components=2)
pca_result = pca.fit_transform(ccTLD_embedding)

print('Explained variation per principal component: {}'.format(pca.explained_variance_ratio_))

plt.figure(figsize=(10, 7))
plt.scatter(pca_result[:,0],pca_result[:,1])
for i in range(num_ccTLD):
    plt.annotate(common.ccTLDs[i],pca_result[i,:])
plt.savefig(args.output+'_pca.png')

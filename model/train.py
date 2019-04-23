#!/bin/python

from __future__ import print_function

def input_fn(dataset):
    return (0,0)

########################################
import argparse
parser=argparse.ArgumentParser('FIXME')
parser.add_argument('--batch_size',type=int,default=32)
parser.add_argument('--shuffle_size',type=int,default=2**14)
parser.add_argument('--vocab_size',type=int,default=2**20)
parser.add_argument('--embedding_size',type=int,default=300)
parser.add_argument('--num_embeddings',type=int,default=5)
parser.add_argument('--nce_samples',type=int,default=2**14)
parser.add_argument('--context_left',type=int,default=3)
parser.add_argument('--context_right',type=int,default=3)
args = parser.parse_args()


########################################
print('loading tensorflow')
import tensorflow as tf

########################################
print('loading vocab')
countries={
    'ar':'argentina',
    'bo':'bolivia',
    'bz':'belize',
    'cl':'chile',
    'co':'colombia',
    'cr':'costa rica',
    'cu':'cuba',
    'do':'dominican republic',
    'ec':'ecuador',
    'sv':'el salvador',
    'gt':'guatemala',
    'hn':'honduras',
    'mx':'mexico',
    'ni':'nicaragua',
    'pa':'panama',
    'py':'paraguay',
    'pe':'peru',
    'pr':'puerto rico',
    'es':'spain',
    'uy':'uruguay',
    've':'venezuela',
    'gq':'equatorial guinea',
}

import pickle
#vocab_filename='bin/all.vocab'
vocab_filename='crawls/ccTLD.ag.jl.vocab'
with open(vocab_filename,'r') as f:
    vocab=pickle.load(f)
    print('  vocab.most_common')
    vocab_top=map(lambda (x,y):x,vocab.most_common(args.vocab_size-1))

########################################
print('creating dataset')

def skipgram_input_fn():
    # create vocab hash
    vocab_index=tf.contrib.lookup.index_table_from_tensor(
        vocab_top,
        default_value=args.vocab_size-1,
        )

    # create skipgram model
    filenames=['crawls/ccTLD.'+ccTLD+'.jl' for ccTLD in countries.keys()]

    dataset_per_filename=[]
    #for filename in filenames:
    for i in range(len(filenames)):
        filename=filenames[i]
        dataset = tf.data.Dataset.from_tensor_slices(filenames)
        dataset = dataset.interleave(
            lambda x: tf.data.TextLineDataset([x]),
            cycle_length=len(filenames),
            block_length=1
            )
        #dataset = tf.data.TextLineDataset(filenames)
        dataset = dataset.map(lambda x: tf.string_split([x],'"').values[7])
        dataset = dataset.map(lambda x: tf.py_func(lambda str: str.lower(),[x],tf.string))
        dataset = dataset.flat_map(lambda x: tf.data.Dataset.from_tensor_slices(tf.string_split([x],'.').values))
        dataset = dataset.map(lambda x: tf.string_split([x],' ').values)
        dataset = dataset.map(lambda x: vocab_index.lookup(x))

        SKIPGRAM_PAD=-2
        def sentence2skipgram_input(sentence):
            sentence_padded=tf.pad(
                sentence,
                tf.constant([[args.context_left,args.context_right]],shape=[1,2]),
                constant_values=SKIPGRAM_PAD,
                )
            context=tf.concat([
                tf.manip.roll(sentence_padded,i,0)
                for i in
                range(-args.context_left,0)+range(1,args.context_right+1)
                ],axis=0)
            words=tf.concat([sentence_padded for i in range(args.context_left+args.context_right)],axis=0)
            return (context,words)
        dataset = dataset.map(sentence2skipgram_input)
        dataset = dataset.flat_map(lambda x,y:tf.data.Dataset.from_tensor_slices(tf.stack([x,y],axis=1)))
        dataset = dataset.filter(lambda x: tf.not_equal(x[1],SKIPGRAM_PAD))
        dataset = dataset.filter(lambda x: tf.not_equal(x[0],SKIPGRAM_PAD))
        dataset = dataset.zip((dataset,tf.data.Dataset.from_tensors([i]).repeat()))
        dataset_per_filename.append(dataset)

    dataset=tf.contrib.data.sample_from_datasets(dataset_per_filename)
    #dataset=dataset.map(lambda x,y: x)

    #dataset = dataset.map(lambda x: (x[0],x[1,tf.newaxis]))
    dataset = dataset.shuffle(args.shuffle_size)
    dataset = dataset.batch(args.batch_size)
    return dataset

#dataset=skipgram_input_fn()
#iterator = dataset.make_initializable_iterator()
#input=iterator.get_next()
#x=input[:,0]
#y=tf.reshape(input[:,1],[-1,1])

########################################
print('graph')

def skipgram_model(
    features, # This is batch_features from input_fn
    labels,   # This is batch_labels from input_fn
    mode,     # An instance of tf.estimator.ModeKeys
    params,   # Additional configuration
    ):

    word=features[:,0]
    context=features[:,1,tf.newaxis]

    import math
    num_ccTLD=len(countries.keys())
    embeddings = tf.get_variable(
        name='embeddings',
        shape=[args.vocab_size,args.embedding_size,args.num_embeddings],
        )
    embeddings_projector = tf.get_variable(
        name='embeddings_projector',
        shape=[args.num_embeddings,num_ccTLD],
        )
    embeddings_ccTLD = tf.tensordot(embeddings,embeddings_projector,[[2],[0]])
    nce_weights = tf.get_variable(
        name='nce_weights',
        shape=[args.vocab_size,args.embedding_size],
        )
    nce_biases = tf.get_variable(
        name='nce_biases',
        shape=[args.vocab_size],
        )

    losses=[]
    for ccTLD in range(num_ccTLD):
        indices=tf.where(tf.equal(labels[:,0],ccTLD))
        word_ccTLD=tf.gather(word,indices)[:,0]
        context_ccTLD=tf.gather(context[:,0],indices)
        embed = tf.nn.embedding_lookup(embeddings_ccTLD, word_ccTLD)
        #embed = tf.Print(embed,[tf.size(word),tf.size(word_ccTLD),tf.size(context_ccTLD)])
        loss_per_dp = tf.nn.nce_loss(
            weights=nce_weights[:,:],
            biases=nce_biases[:],
            labels=context_ccTLD,
            inputs=embed[:,:,ccTLD],
            num_sampled=min(args.nce_samples,args.vocab_size),
            num_classes=args.vocab_size,
            )
        losses.append(loss_per_dp)
    #loss=tf.reduce_mean(losses)
    loss=tf.reduce_sum(map(tf.reduce_sum,losses))/args.batch_size

    #reg=tf.norm(tf.stack([
        #embed[:,:,0:ccTLD],
        #embed[:,:,ccTLD+1:],
        #])-embed[:,:,ccTLD])
    #loss_reg=loss+reg

    if mode==tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(
            mode
            )

    if mode==tf.estimator.ModeKeys.EVAL:
        return tf.estimator.EstimatorSpec(
            mode,
            loss=loss,
            )

    if mode==tf.estimator.ModeKeys.TRAIN:
        learning_rate=tf.train.piecewise_constant(
            tf.train.get_global_step(),
            [int(1e3),int(1e4),int(1e6)],
            [1.0,1e-1,1e-2,1e-3]
            )
        optimizer = tf.train.AdagradOptimizer(learning_rate=learning_rate) #1e-2)
        train_op = optimizer.minimize(loss, global_step=tf.train.get_global_step())
        return tf.estimator.EstimatorSpec(
            mode,
            loss=loss,
            train_op=train_op,
            )

########################################
print('run')

tf.logging.set_verbosity(tf.logging.INFO)
model=tf.estimator.Estimator(
    model_fn=skipgram_model,
    )

model.train(
    input_fn=skipgram_input_fn
    )

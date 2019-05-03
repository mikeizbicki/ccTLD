#!/bin/python

from __future__ import print_function

########################################
import argparse
parser=argparse.ArgumentParser('predict with a skipgram model')
parser.add_argument('--batch_size',type=int,default=32)
parser.add_argument('--model_dir',type=str,required=True)
args = parser.parse_args()

with open(args.model_dir+'/args.json','r') as f:
    import simplejson as json
    data=json.loads(f.readline())
    args_train=type('',(),data)

########################################
print('loading tensorflow')
import tensorflow as tf
import skipgram
import common

#tf.logging.set_verbosity(tf.logging.INFO)
print('creating estimator')
estimator=tf.estimator.Estimator(
    model_fn=skipgram.model_fn_predict,
    model_dir=args.model_dir,
    params=args_train,
    )

print('predicting')
xs=estimator.predict(
    input_fn=lambda: skipgram.input_fn_predict(args_train),
    )

for x in xs:
    #print('x.shape=',x.shape)
    print('word=',x['words'])
    translation=x['translation']
    scores=x['translation_scores']
    for i in range(len(common.ccTLDs)):
        print('%s: '%common.ccTLDs[i],end='')
        for j in range(translation.shape[0]):
            print('%10s (%0.2f) '%(translation[j,i],scores[j,i]),end=' ')
        print()
    print()
    print()

    #for i in range(len(common.ccTLDs)):
        #print('%s: '%common.ccTLDs[i],end='')
        #for j in range(res.shape[0]):
            #print(res[j,i],end=' ')
        #print()

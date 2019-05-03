from __future__ import print_function

def input_fn_predict(args):
    import tensorflow as tf

    # create dataset
    print('  create dataset')
    dataset=tf.data.TextLineDataset(['words.txt'])
    #dataset = dataset.map(lambda x: (vocab_index.lookup(x),x))
    dataset = dataset.batch(args.batch_size)
    return dataset

def model_fn_predict(
    features, # This is batch_features from input_fn
    labels,   # This is batch_labels from input_fn
    mode,     # An instance of tf.estimator.ModeKeys
    params,   # Additional configuration
    ):
    import tensorflow as tf
    import common
    args=params

    # create vocab hash
    print('  create vocab')
    import common
    vocab_top=common.get_vocab(args.vocab_size)
    vocab_index=tf.contrib.lookup.index_table_from_tensor(
        vocab_top,
        default_value=args.vocab_size-1,
        )

    words=features
    features=vocab_index.lookup(words)

    # create variables
    import math
    num_ccTLD=len(common.countries.keys())
    embeddings = tf.get_variable(
        name='embeddings',
        shape=[args.vocab_size,args.embedding_size,args.num_embeddings],
        )
    if args.num_embeddings>1:
        embeddings_projector = tf.get_variable(
            name='embeddings_projector',
            shape=[args.num_embeddings,num_ccTLD],
            )
    else:
        embeddings_projector = tf.ones([args.num_embeddings,num_ccTLD])
    embeddings_ccTLD = tf.tensordot(embeddings,embeddings_projector,[[2],[0]])

    # create word vectors
    wordvecs_raw=tf.gather(
        embeddings,
        features,
        )
    wordvecs=tf.tensordot(
        wordvecs_raw,
        embeddings_projector,
        axes=[[2],[0]],
        )
    embeddings_mean=tf.reduce_mean(wordvecs,axis=2)
    wordvecs_all=tf.tensordot(
        embeddings,
        embeddings_projector,
        axes=[[2],[0]],
        )

    # create word vectors
    ccTLD_source=common.ccTLDs.index('us')
    wordvecs_source=wordvecs[:,:,ccTLD_source]

    # find similar vectors
    res=[]
    for ccTLD in range(0,len(common.ccTLDs)):
        cosine_similarities=tf.tensordot(
            tf.nn.l2_normalize(wordvecs[:,:,ccTLD],axis=1),
            tf.nn.l2_normalize(embeddings_ccTLD[:,:,ccTLD],axis=1),
            axes=[[1],[1]],
            )
        vals,indices=tf.nn.top_k(cosine_similarities,k=10)
        predictions=tf.gather(vocab_top,indices)
        res.append(predictions)
    similar=tf.stack(res,axis=2)

    # measure
    res=[]
    for ccTLD in range(0,len(common.ccTLDs)):
        wordvecs_ccTLD=wordvecs[:,:,ccTLD]
        diff=wordvecs_source-wordvecs_ccTLD
        cosine_similarities=tf.reduce_sum(
            tf.nn.l2_normalize(wordvecs_source,axis=1)*
            tf.nn.l2_normalize(wordvecs_ccTLD,axis=1)
            ,axis=1
            )
        l2=tf.reduce_sum(diff*diff,axis=1)
        l1=tf.reduce_sum(tf.abs(diff),axis=1)
        res.append(tf.stack([cosine_similarities,l2,l1],axis=1))
    measure=tf.stack(res,axis=2)

    # create translation
    res_translation=[]
    res_scores=[]
    for ccTLD in range(0,len(common.ccTLDs)):
        cosine_similarities=tf.tensordot(
            tf.nn.l2_normalize(wordvecs_source,axis=1),
            tf.nn.l2_normalize(wordvecs_all[:,:,ccTLD],axis=1),
            axes=[[1],[1]],
            )
        vals,indices=tf.nn.top_k(cosine_similarities,k=5)
        predictions=tf.gather(vocab_top,indices)
        res_translation.append(predictions)
        res_scores.append(vals)
    translation=tf.stack(res_translation,axis=2)
    translation_scores=tf.stack(res_scores,axis=2)

    # return
    if mode==tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(
            mode,
            predictions={
                'words':words,
                'measure':measure,
                'similar':similar,
                'translation':translation,
                'translation_scores':translation_scores,
                },
            )

################################################################################
def input_fn_train(args):
    """
    WARNING:
    This function contains many subtle hacks designed to ensure that the Unicode
    decoding works and the strings are properly labelled with their country of origin.
    Modify carefully, constantly checking the work with `print_data.py`
    """
    import tensorflow as tf
    with tf.device('/cpu:0'):

        # create vocab hash
        import common
        vocab_top=common.get_vocab(args.vocab_size)
        vocab_top=map(lambda x: x.encode('unicode_escape').decode('unicode_escape'),vocab_top)
        vocab_index=tf.contrib.lookup.index_table_from_tensor(
            vocab_top,
            default_value=args.vocab_size-1,
            )

        # load files
        if args.data_source=='billion':
            filenames=['spanish_billion_words/spanish_billion_words_'+str(i).zfill(2) for i in range(100)]
            compression_type=None
            def json2text(x):
                return x.lower()

        elif args.data_source=='crawls':
            filenames=['crawls/ccTLD.'+ccTLD+'.jl.gz.dedupe' for ccTLD in common.ccTLDs]
            compression_type='GZIP'
            import simplejson as json
            import re
            regex=re.compile(r'[^\w.?!]',re.UNICODE)
            def json2text(x):
                text=json.loads(x)['text'].lower()
                text=regex.sub(' ',text)
                text=text.encode('utf-8').replace('\\n',' ')
                return text

        filenames_index=tf.contrib.lookup.index_table_from_tensor(filenames)

        # extract data from files
        dataset_per_filename=[]
        dataset = tf.data.Dataset.from_tensor_slices(filenames)
        dataset = dataset.interleave(
            lambda x: tf.data.Dataset.zip((
                tf.data.TextLineDataset([x],compression_type=compression_type),
                tf.data.Dataset.from_tensors(filenames_index.lookup(x)).repeat(),
                )),
            cycle_length=len(filenames),
            block_length=1
            )

        # count the number of lines processed
        num_lines=tf.get_variable(
            name='num_lines',
            initializer=0,
            dtype=tf.int32,
            trainable=False,
            collections=[tf.GraphKeys.LOCAL_VARIABLES],
            use_resource=True,
            )
        def update_num_lines(x):
            update_op=tf.assign_add(num_lines,1)
            with tf.control_dependencies([update_op]):
                return x
        dataset = dataset.map(lambda x,y: (update_num_lines(x),y))
        tf.summary.scalar('num_lines',num_lines)

        # parse the json formatted input lines
        dataset = dataset.map(lambda x,y: (tf.py_func(json2text,[x],tf.string),y))
        dataset = dataset.flat_map(lambda x,y: tf.data.Dataset.zip((
            tf.data.Dataset.from_tensor_slices(tf.string_split([x],'.?!').values),
            tf.data.Dataset.from_tensors(y).repeat(),
            )))
        dataset = dataset.map(lambda x,y: (tf.string_split([x],' ').values,y))
        dataset = dataset.map(lambda x,y: (vocab_index.lookup(x),y))

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
        dataset = dataset.map(lambda x,y: (sentence2skipgram_input(x),y))
        dataset = dataset.flat_map(lambda (x,y),z:tf.data.Dataset.zip((
            tf.data.Dataset.from_tensor_slices(tf.stack([x,y],axis=1)),
            tf.data.Dataset.from_tensors(z).repeat(),
            )))

        # remove unuseful data
        dataset = dataset.filter(lambda x,y: tf.not_equal(x[0],SKIPGRAM_PAD))
        dataset = dataset.filter(lambda x,y: tf.not_equal(x[1],SKIPGRAM_PAD))
        #dataset = dataset.filter(lambda x,y: tf.greater_equal(x[0],100))
        #dataset = dataset.filter(lambda x,y: tf.greater_equal(x[1],100))

        # word counter variables
        with tf.variable_scope('all_words'):
            counter_words=tf.get_variable(
                name='counter_words',
                initializer=tf.zeros([args.vocab_size],dtype=tf.int32),
                dtype=tf.int32,
                trainable=False,
                collections=[tf.GraphKeys.LOCAL_VARIABLES],
                use_resource=True,
                )
            counter_total=tf.get_variable(
                name='counter_total',
                initializer=1,
                dtype=tf.int32,
                trainable=False,
                collections=[tf.GraphKeys.LOCAL_VARIABLES],
                use_resource=True,
                )
            counter_total_unfiltered=counter_total
            for p in [0,0.2,0.4,0.6,0.8]:
                start=int(p*args.vocab_size)
                end=int((p+0.2)*args.vocab_size)-1
                word_percent=tf.reduce_mean(tf.cast(counter_words[start:end],tf.float32))/tf.cast(counter_total,tf.float32)
                tf.summary.scalar('rank_%0.2f'%p,word_percent)
            word_percent_unk=tf.cast(counter_words[args.vocab_size-1],tf.float32)/tf.cast(counter_total,tf.float32)
            tf.summary.scalar('rank_UNK',word_percent_unk)
            tf.summary.scalar('counter_total',counter_total)
            def update_counter_words(x):
                update_op1=tf.scatter_update(counter_words,x[0],counter_words[x[0]]+1)
                update_op2=tf.assign_add(counter_total,1)
                with tf.control_dependencies([update_op1,update_op2]):
                    return x
            dataset = dataset.map(lambda x,y: (update_counter_words(x),y))

        # filter counter variables
        with tf.variable_scope('filtered_words'):
            filtered_words=tf.get_variable(
                name='filtered_words',
                initializer=tf.zeros([args.vocab_size],dtype=tf.int32),
                dtype=tf.int32,
                trainable=False,
                collections=[tf.GraphKeys.LOCAL_VARIABLES],
                use_resource=True,
                )
            filtered_total=tf.get_variable(
                name='filtered_total',
                initializer=1,
                dtype=tf.int32,
                trainable=False,
                collections=[tf.GraphKeys.LOCAL_VARIABLES],
                use_resource=True,
                )
            for p in [0,0.2,0.4,0.6,0.8]:
                start=int(p*args.vocab_size)
                end=int((p+0.2)*args.vocab_size)-1
                word_percent=tf.reduce_mean(tf.cast(filtered_words[start:end],tf.float32))/tf.cast(filtered_total,tf.float32)
                tf.summary.scalar('rank_%0.2f_filtered'%p,word_percent)
            word_percent_unk=tf.cast(filtered_words[args.vocab_size-1],tf.float32)/tf.cast(filtered_total,tf.float32)
            tf.summary.scalar('rank_UNK',word_percent_unk)
            tf.summary.scalar('filtered_total',filtered_total)
            tf.summary.scalar('keep_frac',tf.cast(filtered_total,tf.float32)/tf.cast(counter_total,tf.float32))
            def rm_too_popular(x):
                #t=1e-5
                t=10.0/args.vocab_size
                freq=tf.minimum(
                    tf.cast(filtered_words[x[0]],tf.float32)/tf.cast(filtered_total,tf.float32),
                    tf.cast(filtered_words[x[1]],tf.float32)/tf.cast(filtered_total,tf.float32)
                    )
                #p=(f-t)/f-tf.sqrt(t/f)
                p=tf.sqrt(t/freq)
                rand=tf.random_uniform([1])[0]

                def true_and_update():
                    #update_op1=tf.scatter_add(filtered_words,x[0],1)
                    update_op1=tf.scatter_update(filtered_words,x[0],filtered_words[x[0]]+1)
                    update_op2=tf.scatter_update(filtered_words,x[1],filtered_words[x[1]]+1)
                    update_op3=tf.assign_add(filtered_total,1)
                    with tf.control_dependencies([update_op1,update_op2,update_op3]):
                        return True
                #return tf.cond(tf.less_equal(freq,threshold),lambda:True,update_fn)
                ret=tf.cond(
                    #tf.less_equal(freq,threshold),
                    tf.less_equal(rand,p),
                    true_and_update,
                    lambda: False
                    )
                #ret=tf.Print(ret,[filtered_words[x[0]],freq,threshold,ret])
                return ret
        dataset = dataset.filter(lambda x,y: rm_too_popular(x))

        # prep for training
        dataset = dataset.shuffle(args.shuffle_size)
        dataset = dataset.batch(args.batch_size)

    # prefetch onto GPU
    dataset = dataset.prefetch(1)

    return dataset

################################################################################
def model_fn_train(
    features, # This is batch_features from input_fn
    labels,   # This is batch_labels from input_fn
    mode,     # An instance of tf.estimator.ModeKeys
    params,   # Additional configuration
    ):
    import tensorflow as tf
    import common
    args=params

    import math
    num_ccTLD=len(common.ccTLDs)
    embeddings = tf.get_variable(
        name='embeddings',
        shape=[args.vocab_size,args.embedding_size,args.num_embeddings],
        )
    if args.num_embeddings>1:
        embeddings_projector = tf.get_variable(
            name='embeddings_projector',
            shape=[args.num_embeddings,num_ccTLD],
            )
    else:
        embeddings_projector = tf.ones([args.num_embeddings,num_ccTLD])

    nce_weights = tf.get_variable(
        name='nce_weights',
        shape=[args.vocab_size,args.embedding_size],
        )
    nce_biases = tf.get_variable(
        name='nce_biases',
        shape=[args.vocab_size],
        )

    # everything below assumes input_fn_train
    word=features[:,1]
    context=features[:,0,tf.newaxis]

    with tf.variable_scope('loss'):
        losses=[]
        for ccTLD in range(num_ccTLD):
            indices=tf.where(tf.equal(labels[:],ccTLD))
            word_ccTLD=tf.gather(word,indices)[:,0]
            context_ccTLD=tf.gather(context[:,0],indices)

            embeddings_word=tf.gather(
                embeddings,
                word_ccTLD,
                )
            embed=tf.tensordot(
                embeddings_word,
                embeddings_projector[:,ccTLD],
                axes=[[2],[0]],
                )
            loss_per_dp = tf.nn.nce_loss(
                weights=nce_weights[:,:],
                biases=nce_biases[:],
                labels=context_ccTLD,
                inputs=embed,
                num_sampled=min(args.nce_samples,args.vocab_size),
                num_classes=args.vocab_size,
                )
            losses.append(loss_per_dp)
        loss=tf.reduce_sum(map(tf.reduce_sum,losses))/args.batch_size

    with tf.variable_scope('regularization'):
        indices=tf.random_uniform(
            dtype=tf.int64,
            minval=0,
            maxval=args.vocab_size,
            shape=[min(args.nce_samples,args.vocab_size)],
            )

        embeddings_sample=tf.gather(
            embeddings,
            indices,
            )

        embeddings_sample_mean=tf.reduce_mean(
            embeddings_sample,
            axis=2,
            )

        regs=[]
        for i in range(args.num_embeddings):
            embeddings_i=embeddings_sample[:,:,i]
            diff=embeddings_sample_mean-embeddings_i

            if args.reg_l2_diff > 0:
                regs.append(args.reg_l2_diff*tf.reduce_sum(diff*diff))

            if args.reg_l1_diff > 0:
                regs.append(args.reg_l1_diff*tf.reduce_sum(tf.abs(diff)))

        null_pt=tf.concat([
            tf.ones([1]),
            tf.zeros([args.embedding_size-1]),
            ],axis=0)
        mean_reg_pt=embeddings_sample_mean-null_pt
        if args.reg_l2_mean > 0:
            regs.append(args.reg_l2_mean*tf.reduce_sum(mean_reg_pt*mean_reg_pt))

        if args.reg_l1_mean > 0:
            regs.append(args.reg_l1_mean*tf.reduce_sum(tf.abs(mean_reg_pt)))

        reg=tf.reduce_mean(regs)

    loss_regularized=loss+reg

    if mode==tf.estimator.ModeKeys.EVAL:
        return tf.estimator.EstimatorSpec(
            mode,
            loss=loss,
            )

    if mode==tf.estimator.ModeKeys.TRAIN:
        #learning_rate=tf.train.piecewise_constant(
            #tf.train.get_global_step(),
            #[int(1e3),int(1e4),int(1e6)],
            #[1.0,1e-1,1e-2,1e-3],
            #)
        #optimizer = tf.train.AdagradOptimizer(learning_rate=learning_rate) #1e-2)
        learning_rate=5e-4
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
        #optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
        train_op = optimizer.minimize(loss_regularized, global_step=tf.train.get_global_step())
        return tf.estimator.EstimatorSpec(
            mode,
            loss=loss,
            train_op=train_op,
            )


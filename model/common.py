countries={
    'ar':'Argentina',
    'bo':'Bolivia',
    'cl':'Chile',
    'co':'Colombia',
    'cr':'Costa Rica',
    'cu':'Cuba',
    'do':'Dominican Republic',
    'ec':'Ecuador',
    'sv':'El Salvador',
    'gt':'Guatemala',
    'hn':'Honduras',
    'mx':'Mexico',
    'ni':'Nicaragua',
    'pa':'Panama',
    'py':'Paraguay',
    'pe':'Peru',
    'pr':'Puerto Rico',
    'es':'Spain',
    'uy':'Uruguay',
    've':'Venezuela',
    'gq':'Equatorial Guinea',

    'us':'United States',
    'pt':'Portugal',
    'bz':'Belize',
    'br':'Brazil',
}

ccTLDs=sorted(countries.keys())

def get_vocab(vocab_size,vocab_filename='bin/all.vocab'):
    import pickle
    vocab_top_filename='bin/'+str(vocab_size)+'.vocab'
    try:
        with open(vocab_top_filename,'r') as f:
            vocab_top=pickle.load(f)
    except:
        with open(vocab_filename,'r') as f:
            vocab=pickle.load(f)
            vocab_top=map(lambda (x,y):x,vocab.most_common(vocab_size-1))
            vocab_top.append('<<UNK>>')
        with open(vocab_top_filename,'w') as f:
            pickle.dump(vocab_top,f)
    return vocab_top


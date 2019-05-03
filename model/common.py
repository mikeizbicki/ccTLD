countries={
    'ar':'argentina',
    'bo':'bolivia',
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

    'us':'united states',
    'pt':'portugal',
    'bz':'belize',
    'br':'brazil',
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


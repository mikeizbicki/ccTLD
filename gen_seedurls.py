from googlesearch import search
import time

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

for tld in countries.keys():
    with open('seedurls2/'+tld,'w') as f:
        for url in search(countries[tld]+' site:.'+tld, stop=500):
            f.write(url+'\n')
            print(url)
            time.sleep(10)

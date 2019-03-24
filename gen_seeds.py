from googlesearch import search
import time

ccTLDs=['ar','bo','cl','co','cr','cu','do','ec','sv','gt','hn','mx','ni','pa','py','pe','pr','es','uy','ve','gq']

for tld in ccTLDs:
    with open('seedurls2/'+tld,'w') as f:
        for url in search('site:.'+tld, stop=500):
            f.write(url+'\n')
            print(url)
            time.sleep(10)

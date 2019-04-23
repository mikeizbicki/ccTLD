#!/usr/bin/python
# -*- coding: utf-8 -*-

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

# FIXME: 'ad':'andora'
# FIXME: 'cat':'catalan'
# FIXME: 'eus':'euskera'
# FIXME: 'gal':'gallega'
# 'as':'Samoa Americana'
# 'gu':'Guam'
countries={
    'ph':'filipinas',
    'us':'estados unidos',
    'ca.us':'california',
    'az.us':'arizona',
    'nm.us':'new mexico',
    'tx.us':'texas',
    'fl.us':'florida',

    'gq':'equatorial guinea',
    'br':'brazil',
    'pt':'portugal',
    'bs':'bahamas',
    'ai':'anguilla',
    'ag':'antigua y barbuda',
    'gd':'grenada',
    'gy':'guyana',
    'jm':'jamaica',
    'ms':'montserrat',
    'kn':'san cristobal y nieves',
    'gs':'islas georgias del sur y sandwich del sur',
    'tt':'trinidad y tobago',
    'tc':'turcs y caicos',
    'vg':'islas virgenes britanicas',
    'vi':'islas virgenes de los estados unidos',

    'lat':'latin america',
    'rio':'rio de janeiro',
    'vuelos':'vuelos',
    'viajes':'viajes',
    'uno':'uno',
    'tienda':'tienda',
    'soy':'soy',
    'ltda':'empresas',
    'juegos':'juegos',
    'hoteles':'hoteles',
    'futbol':'futbol',
    'gratis':'gratis',
    'abogado':'abogado',
    }

for tld in countries.keys():
    with open('seedurls/'+tld,'w') as f:
        for url in search(countries[tld]+' site:.'+tld, stop=500):
            f.write(url+'\n')
            print(url)
            time.sleep(10)

phrases=['noticias','universidad','gobierno']
for tld in countries.keys():
    with open('seedurls/'+tld,'a') as f:
        for phrase in phrases:
            for url in search(phrase+' site:.'+tld, stop=500):
                f.write(url+'\n')
                print(url)
                time.sleep(10)

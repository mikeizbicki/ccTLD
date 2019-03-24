import scrapy
import urlextract
import re
import logging
from bs4 import BeautifulSoup
import newspaper
import goose3
import langdetect 
from urllib.parse import urlparse


#ccTLDs=['ar','bo','cl','co','cr','cu','do','ec','sv','gt','hn','mx','ni','pa','py','pe','pr','es','uy','ve','gq']

class body(scrapy.Spider):
    name = "body"

    def __init__(self, cc=None, language='es', **kwargs):

        self.g=goose3.Goose()
        self.language=language

        # get ccTLDs from `-a` cmd line arg 
        if cc is None:
            self.allowed_domains = ccTLDs
        else:
            self.allowed_domains = [cc]

        # load seed urls
        self.start_urls=[]
        for domain in self.allowed_domains:
            with open('seedurls/'+domain) as f:
                line=f.readline()
                while line:
                    self.start_urls.append(line)
                    line=f.readline()

        # recursively init
        super().__init__(**kwargs)

    def parse(self, response):
        response_tld=urlparse(response.url).hostname.split('.')[-1]
        self.crawler.stats.inc_value('responses')
        self.crawler.stats.inc_value('responses_'+response_tld)

        # process text responses
        if isinstance(response,scrapy.http.TextResponse):
            soup = BeautifulSoup(response.text)
            lang = langdetect.detect(soup.text)

            self.crawler.stats.inc_value('text')
            self.crawler.stats.inc_value('text_'+response_tld)

            # only process webpages in the selected language
            if lang==self.language:

                self.crawler.stats.inc_value('lang')
                self.crawler.stats.inc_value('lang_'+response_tld)

                # if webpage has major content,
                # then yield this content
                article=self.g.extract(raw_html=response.text)
                text=article.cleaned_text
                if len(text)>10:
                    self.crawler.stats.inc_value('yield')
                    self.crawler.stats.inc_value('yield_'+response_tld)
                    yield {
                            'url':response.url,
                            'text':text,
                    }

                # calculate priorities
                counts={}
                for tld in self.allowed_domains:
                    counts[tld]=self.crawler.stats.get_value('yield_'+tld)
                    if counts[tld] is None:
                        counts[tld]=0
                priorityList={key: rank for rank, key in enumerate(sorted(counts, key=counts.get, reverse=True))}
                priority=priorityList[tld]

                # find urls within html tags
                # these urls may be relative urls (i.e. no TLD)
                # FIXME: are there more html tags that should be added to the list?
                for url in response.css('a::attr(href)'):
                    url_str=url.get()
                    if not re.match(r'javascript:',url_str):
                        yield response.follow(url_str, callback=self.parse, priority=priority)
                        self.crawler.stats.inc_value('url')
                        self.crawler.stats.inc_value('url_'+response_tld)
                        #str='url_'+tld
                        #print(str,':',self.crawler.stats.get_value(str))

                if len(text)>10:
                    responses={}
                    for tld in self.allowed_domains:
                        responses[tld]=self.crawler.stats.get_value('responses_'+tld)
                        if responses[tld] is None:
                            responses[tld]=0
                    urls={}
                    for tld in self.allowed_domains:
                        urls[tld]=self.crawler.stats.get_value('url_'+tld)
                        if urls[tld] is None:
                            urls[tld]=0
                    #print('responses=',responses)
                    #print('counts=',counts)
                    #print('priorityList=',priorityList)

                    print()
                    for tld in self.allowed_domains:
                        print(tld+': %10d %10d %10d %10d'%(responses[tld],urls[tld],counts[tld],priorityList[tld]))


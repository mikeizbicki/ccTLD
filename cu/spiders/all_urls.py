import scrapy
import urlextract
import re
import logging

class all_urls(scrapy.Spider):
    name = "all_urls"
    #allowed_domains = ['kp']
    #start_urls = ['http://www.dprkportal.kp']
    #start_urls = ['http://www.ryongnamsan.edu.kp/univ/switchlang?lang=en']
    allowed_domains = ['cu']
    start_urls = ['http://www.cuba.cu/']

    def parse(self, response):
        urlextractor=urlextract.URLExtract()

        #logging.log(logging.DEBUG, "url="+response.url)

        # process text responses
        if isinstance(response,scrapy.http.TextResponse):

            # find all absolute urls (i.e. urls containing TLDs)
            # these urls need not be in html tags
            urls=urlextractor.find_urls(response.text,only_unique=True)
            #print('urls=',urls)
            for url in urls:

                # css element false positives
                if '.' == url[0]:
                    pass

                # found email 
                elif '@' in url:
                    pass

                # found naked url, need to add protocol
                elif not re.match(r'^http',url):
                    yield response.follow('http://'+url,self.parse)
                    yield response.follow('https://'+url,self.parse)

                # found well-formed url
                else:
                    yield response.follow(url,self.parse)

            # find urls within html tags
            # these urls may be relative urls (i.e. no TLD)
            # FIXME: are there more html tags that should be added to the list?
            for urls in [response.css('link::attr(href)'),
                         response.css('a::attr(href)'),
                         response.css('img::attr(src)'),
                         response.css('script::attr(src)'),
                         response.css('frame::attr(src)'),
                         response.css('embed::attr(src)'),
                         response.css('iframe::attr(src)'),
                         response.css('source::attr(src)'),
                         response.css('object::attr(data)'),
                         ]:
                for url in urls:
                    url_str=url.get()
                    if not re.match(r'javascript:',url_str):
                        yield response.follow(url_str, callback=self.parse)

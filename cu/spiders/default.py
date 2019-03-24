# -*- coding: utf-8 -*-
import scrapy
from scrapy.spiders import CrawlSpider, Rule
from scrapy.linkextractors import LinkExtractor

class DefaultSpider(CrawlSpider):
    name = 'default'
    allowed_domains = ['cu']
    #start_urls = ['http://www.cuba.cu/']
    start_urls = ['http://www.dprkportal.kp']

    rules = [
        Rule(
            LinkExtractor(
                canonicalize=True,
                unique=True
            ),
            follow=True,
            callback="parse_url"
        )
        ]

    def parse_url(self, response):
        pass
        #print('url=',response.url)

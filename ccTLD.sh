#!/bin/sh

if [ $1 = '' ]; then
    echo '$1 must be a ccTLD'
    exit
fi

mkdir -p crawls

scrapy crawl cctld -a cc=$1 -o crawls/ccTLD.${1}.jl -s JOBDIR=crawls/ccTLD.${1}.jobdir

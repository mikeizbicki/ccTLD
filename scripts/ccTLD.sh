#!/bin/sh

export PYTHONUNBUFFERED=true
if [ "$SLURM_SUBMIT_DIR" != '' ]; then
    cd $SLURM_SUBMIT_DIR
    date
    hostname
    echo "\$1=$1"
    scrapy=~/.conda/envs/py3/bin/scrapy
else
    scrapy=scrapy
    scrapy=~/.conda/envs/py3/bin/scrapy
fi

if [ $1 = '' ]; then
    echo '$1 must be a ccTLD'
    exit
fi

mkdir -p crawls

$scrapy crawl ccTLD -a cc=$1 -o crawls/ccTLD.${1}.jl -s JOBDIR=crawls/ccTLD.${1}.jobdir

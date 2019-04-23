#!/bin/sh

python -u src/make_vocab.py --filename=$1

echo
python -u src/analyze_vocab.py --filename=${1}.vocab

echo
python -u src/analyze_vocab.py --filename=${1}.domain

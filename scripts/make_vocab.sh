#!/bin/sh

python -u src/dedupe.py --filename=$1

echo
python -u src/make_vocab.py --filename=${1}.dedupe

echo
python -u src/analyze_vocab.py --filename=${1}.dedupe.vocab

echo
python -u src/analyze_vocab.py --filename=${1}.dedupe.domain

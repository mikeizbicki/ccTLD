#!/bin/sh

#SBATCH --mem=10G

python -u src/merge_vocab.py --filename crawls/*dedupe.vocab --output=./all.vocab
python -u src/analyze_vocab.py --filename=all.vocab --outputdict=all.vocab.dict


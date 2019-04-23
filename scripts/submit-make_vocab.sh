#!/bin/sh

for f in crawls/*.jl; do
    sbatch scripts/make_vocab.sh $f
done

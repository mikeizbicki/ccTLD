#!/bin/sh

for f in crawls/*.jl.gz; do
    sbatch --output=slurm/make_vocab.$(basename $f).out scripts/make_vocab.sh $f
done

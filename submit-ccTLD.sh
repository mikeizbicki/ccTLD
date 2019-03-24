#!/bin/bash

mkdir -p crawls
for f in seedurls/*; do
    tld=$(basename $f)
    sbatch --time=21-00:00:00 --output=crawls/ccTLD.${tld}.slurm ./ccTLD.sh $tld
done

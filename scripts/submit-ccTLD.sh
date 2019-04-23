#!/bin/bash

mkdir -p crawls
for f in seedurls.new/*; do
#for f in seedurls/co seedurls/cr; do
    tld=$(basename $f)
    sbatch --time=21-00:00:00 --mem=6g --output=crawls/ccTLD.${tld}.slurm --job-name="ccTLD.sh $tld" scripts/ccTLD.sh $tld
done

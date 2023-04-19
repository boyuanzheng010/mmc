#!/bin/bash

set -e

annotators="JTWilson JamiesonAlexander BarryAdkins"

python convert-uds-disagreements-to-tiebreaker-hit-input.py \
    uds-train-{disagreements-1.csv,tiebreaker-hits-assignments-2.jsonl} \
    --redundancy 1 \
    --tolerance 10 \
    --annotators $annotators
python split-assignments.py uds-train-tiebreaker-hits-assignments-2{.jsonl,}
for a in $annotators
do
    python convert-jsonl-csv.py uds-train-tiebreaker-hits-assignments-2/${a}.{jsonl,csv}
done

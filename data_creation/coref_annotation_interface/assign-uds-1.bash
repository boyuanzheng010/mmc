#!/bin/bash

set -e

annotators="JTWilson JoselynCarretero GusRodriguez NataliaMauroni JamiesonAlexander"

function assign {
    local split="$1"
    local redundancy="$2"
    python make-assignments.py uds-${split}-hits{,-assignments-1}.jsonl \
        --redundancy $redundancy \
        --annotators $annotators
    python split-assignments.py uds-${split}-hits-assignments-1{.jsonl,}
}

assign train 2
assign dev 3
assign test 3

mkdir -p uds-combined-hits-assignments-1{,-shuf}
for a in $annotators
do
    cat uds-{train,dev,test}-hits-assignments-1/${a}.jsonl > uds-combined-hits-assignments-1/${a}.jsonl
    python shuffle-across-docs.py uds-combined-hits-assignments-1{,-shuf}/${a}.jsonl
    python convert-jsonl-csv.py uds-combined-hits-assignments-1-shuf/${a}.{jsonl,csv}
done

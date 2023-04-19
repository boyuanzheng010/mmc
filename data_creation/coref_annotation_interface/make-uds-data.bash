#!/bin/bash

set -e

parse_root="/export/corpora/LDC/LDC2012T13/eng_web_tbk/data"

for split in train dev test
do
    python extract-uds-spans.py \
        --split ${split} \
        --query-spans 3pp \
        --candidate-syntax-spans limited \
        --parse-root $parse_root \
        --fix-leaf-text \
        uds-${split}.jsonl
    python convert-uds-spans-to-hit-input.py \
        --min-context-size 0 \
        uds-${split}{,-hits}.jsonl
done

#!/usr/bin/env python3

from collections import defaultdict
from random import shuffle

from coref import read_jsonl, write_jsonl


def shuffle_across_docs(hits):
    docs = defaultdict(list)
    for hit in hits:
        docs[hit['documentId']].append(hit)
    docs = list(docs.values())

    shuffle(docs)

    for doc in docs:
        for hit in doc:
            yield hit


def main():
    from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
    parser = ArgumentParser(
        description='Shuffle HITs across (but not within) docs.',
        formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument('input_path', help='Path to input jsonl HIT data.')
    parser.add_argument('output_path', help='Path to output jsonl HIT data.')
    args = parser.parse_args()

    write_jsonl(
        shuffle_across_docs(read_jsonl(args.input_path)),
        args.output_path)


if __name__ == '__main__':
    main()

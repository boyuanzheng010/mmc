#!/usr/bin/env python3

import logging

from coref import read_jsonl, write_jsonl, iter_sentences_as_docs


LOGGER = logging.getLogger(__name__)


def main():
    from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
    parser = ArgumentParser(
        description='Convert jsonl-formatted sentences to jsonl-formatted documents.',
        formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument('input_path', help='Path to input (sentence-level) .jsonl file.')
    parser.add_argument('output_path', help='Path to output (document-level) .jsonl file.')
    args = parser.parse_args()

    LOGGER.setLevel(logging.INFO)
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter('%(asctime)-15s %(levelname)-8s %(message)s'))
    LOGGER.addHandler(handler)

    LOGGER.info('sentences in {} -> documents in {}'.format(args.input_path, args.output_path))
    write_jsonl(iter_sentences_as_docs(read_jsonl(args.input_path)), args.output_path)


if __name__ == '__main__':
    main()

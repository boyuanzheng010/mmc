#!/usr/bin/env python3

from coref import read_jsonl, write_conll_predictions


def main():
    from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
    parser = ArgumentParser(
        description='Convert jsonl predictions to CoNLL format format.',
        formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument('input_path', help='Path to predictions in json lines format.')
    parser.add_argument('output_path',
                        help='Path where predictions will be written in CoNLL format.')
    args = parser.parse_args()
    write_conll_predictions(read_jsonl(args.input_path), args.output_path)


if __name__ == '__main__':
    main()

#!/usr/bin/env python3

from collections import Counter

from coref import read_jsonl, human_format_span


def main():
    from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
    parser = ArgumentParser(
        description='Explore document spans in json lines format.',
        formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument('input_path', help='Path to documents in json lines format')
    parser.add_argument('--inline', action='store_true', help='Show spans inline using brackets')
    parser.add_argument('--span-key', default='spans', help='Show spans from this key')
    args = parser.parse_args()

    for doc in read_jsonl(args.input_path):
        print()
        print(doc['document_id'])
        if args.inline:
            for sentence in doc['sentences']:
                span_starts = Counter(s[0] for s in sentence[args.span_key])
                span_ends = Counter(s[1] for s in sentence[args.span_key])
                print(' '.join(
                    '{}{}{}'.format(
                        '[' * span_starts.get(token_num, 0),
                        token,
                        ']' * span_ends.get(token_num + 1, 0))
                    for (token_num, token) in enumerate(sentence['tokens'])))
        else:
            for sentence in doc['sentences']:
                print(' '.join(sentence['tokens']))
                for span in sentence[args.span_key]:
                    print('*', human_format_span(span, sentence['tokens']))


if __name__ == '__main__':
    main()

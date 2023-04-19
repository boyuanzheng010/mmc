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
        print(doc['documentId'])
        if args.inline:
            for (sentence_num, sentence) in enumerate(doc['sentences']):
                span_starts = Counter(s['startToken'] for s in doc[args.span_key] if s['sentenceIndex'] == sentence_num)
                span_ends = Counter(s['endToken'] for s in doc[args.span_key] if s['sentenceIndex'] == sentence_num)
                print(' '.join(
                    '{}{}{}'.format(
                        '[' * span_starts.get(token_num, 0),
                        token,
                        ']' * span_ends.get(token_num + 1, 0))
                    for (token_num, token) in enumerate(sentence)))
        else:
            for (sentence_num, sentence) in enumerate(doc['sentences']):
                print(' '.join(sentence))
                for span in doc[args.span_key]:
                    if span['sentenceIndex'] == sentence_num:
                        print('*', human_format_span((span['startToken'], span['endToken']), sentence))


if __name__ == '__main__':
    main()

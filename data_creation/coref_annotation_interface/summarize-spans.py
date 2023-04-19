#!/usr/bin/env python3

from collections import Counter

from coref import read_jsonl, human_format_span


def main():
    from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
    parser = ArgumentParser(
        description='Summarize document spans in json lines format.',
        formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument('input_path', help='Path to documents in json lines format')
    parser.add_argument('--span-key', default='spans', help='Show spans from this key')
    parser.add_argument('--num-top-spans', type=int, default=100,
                        help='Number of top spans to show')
    args = parser.parse_args()

    span_counter = Counter()
    for doc in read_jsonl(args.input_path):
        for sentence in doc['sentences']:
            for span in sentence[args.span_key]:
                span_words = tuple(w.lower() for w in sentence['tokens'][span[0]:span[1]])
                span_counter[span_words] += 1

    for (span_words, span_count) in span_counter.most_common(args.num_top_spans):
        print('{:>6} {}'.format(span_count, ' '.join(span_words)))


if __name__ == '__main__':
    main()

#!/usr/bin/env python

import argparse
import csv
import json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('input_path', help='Path to input CSV file.')
    parser.add_argument('output_path', help='Path to output CSV file.')
    parser.add_argument('--input-key', default='Answer.answer_spans',
                        help="Name of input CSV column containing answer span lists")
    parser.add_argument('--output-query-key', default='Answer.query_span',
                        help="Name of output CSV column containing individual query span")
    parser.add_argument('--output-answer-key', default='Answer.answer_span',
                        help="Name of output CSV column containing individual answer span")
    args = parser.parse_args()

    with open(args.input_path, encoding='utf-8') as in_f, \
            open(args.output_path, encoding='utf-8', mode='w') as out_f:
        writer = None
        for row in csv.DictReader(in_f):
            answer_spans = json.loads(row.pop(args.input_key))
            for answer_span in answer_spans:
                query_span = answer_span.pop('querySpan')
                row[args.output_query_key] = '{} {} {}'.format(
                    query_span['sentenceIndex'], query_span['startToken'], query_span['endToken'])
                row[args.output_answer_key] = '{} {} {} {}'.format(
                    answer_span['sentenceIndex'], answer_span['startToken'], answer_span['endToken'],
                    1 if answer_span['notPresent'] else 0)
                if writer is None:
                    writer = csv.DictWriter(out_f, fieldnames=row.keys())
                    writer.writeheader()
                writer.writerow(row)


if __name__ == "__main__":
    main()

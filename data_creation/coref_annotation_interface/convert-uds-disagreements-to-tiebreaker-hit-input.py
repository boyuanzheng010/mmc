#!/usr/bin/env python3

import logging
import json
from csv import DictReader

from coref import write_jsonl, make_assignments


DEFAULT_REDUNDANCY = 1
DEFAULT_TOLERANCE = 10


def convert_disagreements(disagreements, annotators,
                          redundancy=DEFAULT_REDUNDANCY, tolerance=DEFAULT_TOLERANCE):
    def _iter():
        num_disagreements = 0
        for disagreement in disagreements:
            input_data = json.loads(disagreement['input_json'])
            query_answer_data = json.loads(disagreement['query_answer_json'])
            yield dict(
                doc=[dict(
                    documentId=input_data['documentId'],
                    startSentenceIndex=input_data['startSentenceIndex'],
                    endSentenceIndex=input_data['endSentenceIndex'],
                    sentenceIds=input_data['sentenceIds'],
                    sentences=input_data['sentences'],
                    querySpan=query_answer_data['querySpan'],
                    candidateSpans=[
                        dict((k, v) for (k, v) in answer_span.items() if k != 'annotator')
                        for answer_span in query_answer_data['answerSpans']
                    ],
                )],
                assignments=[[
                    answer_span['annotator'] for answer_span in query_answer_data['answerSpans']
                ]]
            )
            num_disagreements += 1

        logging.info('{} HITs processed'.format(num_disagreements))

    return make_assignments(
        _iter(),
        annotators=annotators,
        redundancy=redundancy,
        tolerance=tolerance)


def main():
    from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
    parser = ArgumentParser(
        description='Convert disagreements extracted from UDS into tiebreaker'
                    ' HITs with assignments.',
        formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument('input_path', help='Path to input csv file.')
    parser.add_argument('output_path', help='Path to output jsonl file.')
    parser.add_argument('--annotators', nargs='+', help='Names of annotators.')
    parser.add_argument('--redundancy', type=int, default=DEFAULT_REDUNDANCY,
                        help='Number of times to annotate each item.')
    parser.add_argument('--tolerance', type=int, default=DEFAULT_TOLERANCE,
                        help='Allowable difference in number of items assigned to each annotator.')
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s %(name)-12s %(levelname)-8s %(message)s')

    with open(args.input_path) as f:
        write_jsonl(
            convert_disagreements(DictReader(f),
                                  annotators=args.annotators,
                                  redundancy=args.redundancy,
                                  tolerance=args.tolerance),
            args.output_path)


if __name__ == '__main__':
    main()

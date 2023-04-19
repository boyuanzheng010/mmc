#!/usr/bin/env python3

import csv
import json
from collections import defaultdict, Counter


def freeze(answer):
    return json.dumps(answer)


def unfreeze(frozen_answer):
    return json.loads(frozen_answer)


def main():
    from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
    parser = ArgumentParser(
        description='Summarize HIT results from CSV format.',
        formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument('input_path', help='Path to results in CSV format')
    parser.add_argument('--answer-key', default='Answer.answer_spans',
                        help='Name of CSV column containing JSON-encoded answer')
    parser.add_argument('--input-key', default='Input.json_data',
                        help='Name of CSV column containing JSON-encoded input')
    parser.add_argument('--max-line-width', type=int, default=80,
                        help='Maximum number of characters per line in answer output')
    args = parser.parse_args()

    hit_answers = defaultdict(dict)
    hit_docs = dict()
    with open(args.input_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            hit_id = row['HITId']
            worker_id = row['WorkerId']
            doc = json.loads(row[args.input_key])
            answer = json.loads(row[args.answer_key])

            hit_answers[hit_id][worker_id] = answer
            hit_docs[hit_id] = doc

    hit_ids = sorted(hit_answers)
    hit_query_spans = dict()

    for hit_id in hit_ids:
        worker_answers = hit_answers[hit_id]
        query_span_answers = defaultdict(list)
        query_spans = None
        for (worker_id, answer) in worker_answers.items():
            if query_spans is None:
                query_spans = [freeze(answer_span['querySpan']) for answer_span in answer]
            for answer_span in answer:
                query_span_answers[freeze(answer_span['querySpan'])].append(answer_span)

        hit_query_spans[hit_id] = query_spans

        doc = hit_docs[hit_id]
        sentences = doc['sentences']
        for (query_num, frozen_query_span) in enumerate(query_spans):
            answer_spans = query_span_answers[frozen_query_span]
            query_span = unfreeze(frozen_query_span)
            query_text = ' '.join(
                sentences[query_span['sentenceIndex']][query_span['startToken']:query_span['endToken']])
            tokens = []
            print()
            print('{:>20}  {:>2}  {}'.format(hit_id, query_num, query_text))
            for (sentence_num, sentence) in enumerate(sentences):
                span_starts = Counter(s['startToken'] for s in answer_spans if s['sentenceIndex'] == sentence_num)
                span_ends = Counter(s['endToken'] for s in answer_spans if s['sentenceIndex'] == sentence_num)
                tokens += [
                    '{}{}{}'.format(
                        ('[' * span_starts.get(token_num, 0)) + (
                            '<'
                            if query_span['sentenceIndex'] == sentence_num and query_span['startToken'] == token_num
                            else ''),
                        token,
                        (
                            '>'
                            if query_span['sentenceIndex'] == sentence_num and query_span['startToken'] == token_num
                            else ''
                        ) + (']' * span_ends.get(token_num + 1, 0)))
                    for (token_num, token) in enumerate(sentence)]

            line = ''
            for token in tokens:
                if line:
                    line_with_token = line + ' ' + token
                    if len(line_with_token) <= args.max_line_width:
                        line = line_with_token
                    else:
                        print(line)
                        line = token
                else:
                    line = token
            if line:
                print(line)

            for answer_span in answer_spans:
                if answer_span['notPresent']:
                    print('X', end='')
            print()

    worker_agreement_counts = defaultdict(dict)
    for (hit_id, worker_answers) in hit_answers.items():
        # Query spans are embedded in answer spans so no need to differentiate explicitly
        answer_span_counts = Counter(
            freeze(answer_span)
            for answer in worker_answers.values()
            for answer_span in answer)
        for (worker_id, answer) in worker_answers.items():
            for answer_span in answer:
                key = (hit_id, freeze(answer_span['querySpan']))
                worker_agreement_counts[worker_id][key] = answer_span_counts[freeze(answer_span)] - 1

    print()
    print('{:>20}  {}'.format('Worker ID', 'No. agreements on each query'))
    for (worker_id, hit_agreement_counts) in sorted(worker_agreement_counts.items(),
                                                    key=lambda p: -len(p[1])):
        print('{:>20}  {}'.format(
            worker_id,
            '  '.join('{:>2}'.format(
                hit_agreement_counts.get((hit_id, frozen_query_span), ' '))
                      for hit_id in hit_ids
                      for frozen_query_span in hit_query_spans[hit_id])))


if __name__ == '__main__':
    main()

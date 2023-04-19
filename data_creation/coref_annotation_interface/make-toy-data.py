#!/usr/bin/env python

import json


START_BRACKET_TYPES = {
    '[': 'query',
    '<': 'candidate'
}

END_BRACKET_TYPES = {
    ']': 'query',
    '>': 'candidate'
}


def parse_input_data(input_path):
    sentences = []
    spans = dict(query=[], candidate=[])

    with open(input_path, encoding='utf-8') as f:
        for (sentence_num, line) in enumerate(f):
            tokens = line.strip().split()
            filtered_tokens = []
            span_starts = dict(query=[], candidate=[])
            for (token_num, token) in enumerate(tokens):
                while token and token[0] in START_BRACKET_TYPES:
                    span_type = START_BRACKET_TYPES[token[0]]
                    span_starts[span_type].append(dict(sentenceIndex=sentence_num,
                                                       startToken=token_num))
                    token = token[1:]
                while token and token[-1] in END_BRACKET_TYPES:
                    span_type = END_BRACKET_TYPES[token[-1]]
                    span = span_starts[span_type].pop()
                    span['endToken'] = token_num + 1
                    spans[span_type].append(span)
                    token = token[:-1]
                filtered_tokens.append(token)

            sentences.append(filtered_tokens)
            if span_starts['query'] or span_starts['candidate']:
                raise Exception('not all spans closed by end of sentence')

    return (sentences, spans['query'], spans['candidate'])


def create_toy_data_single_query(sentences, query_spans, output_path):
    with open(output_path, mode='w') as f:
        for query_span in query_spans:
            data = dict(sentences=sentences,
                        querySpan=query_span)
            f.write(json.dumps(data) + '\n')


def create_toy_data_multi_query(sentences, query_spans, output_path):
    with open(output_path, mode='w') as f:
        data = dict(sentences=sentences,
                    querySpans=query_spans)
        f.write(json.dumps(data) + '\n')


def create_toy_data_multi_query_constrained(sentences, query_spans, candidate_spans,
                                            output_path):
    with open(output_path, mode='w') as f:
        data = dict(sentences=sentences,
                    querySpans=query_spans,
                    candidateSpans=candidate_spans)
        f.write(json.dumps(data) + '\n')


def main():
    from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
    parser = ArgumentParser(
        formatter_class=ArgumentDefaultsHelpFormatter,
        description='Convert toy data text to JSON lines format.'
    )
    parser.add_argument('input_path',
                        help='Path to input text file: one sentence per line with '
                             'space-separated tokens, square brackets enclosing query '
                             'spans, and angle brackets enclosing allowed candidate '
                             'spans, for example: <the [<block <party>>>] was fun .')
    parser.add_argument('output_path', help='Path to output JSON lines file.')
    parser.add_argument('--format', choices=('single', 'multi', 'multi-constrained'),
                        default='multi-constrained',
                        help='Whether to output single-query or multi-query format.')
    args = parser.parse_args()

    (sentences, query_spans, candidate_spans) = parse_input_data(args.input_path)
    if args.format == 'single':
        create_toy_data_single_query(sentences, query_spans, args.output_path)
    elif args.format == 'multi':
        create_toy_data_multi_query(sentences, query_spans, args.output_path)
    elif args.format == 'multi-constrained':
        create_toy_data_multi_query_constrained(sentences, query_spans, candidate_spans,
                                                args.output_path)
    else:
        raise Exception('unknown output format {}'.format(args.format))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3

import logging

QUERY_PRONOUN_SPAN_WORDS = {
    'they',
    'them',
    'themself',
    'themselves',
    'their',
    'theirs',
    'she',
    'her',
    'herself',
    'hers',
    'he',
    'him',
    'himself',
    'his',
    'you',
    'your',
    'yours',
    'yourself',
    'yourselves',
    'we',
    'us',
    'our',
    'ours',
    'ourself',
    'ourselves',
    'i',
    'me',
    'my',
    'myself',
    'mine',
}

QUERY_SPAN_START = '['
QUERY_SPAN_END = ']'
CANDIDATE_SPAN_START = '<'
CANDIDATE_SPAN_END = '>'


def add_query_spans_to_toy_data_input(input_path, output_path,
                                      query_span_words=QUERY_PRONOUN_SPAN_WORDS):
    sentences = []
    with open(input_path, encoding='utf-8') as f:
        for line in f:
            sentences.append(line.strip().split())

    logging.info(f'Read {len(sentences)} sentences, {sum(len(s) for s in sentences)} tokens')

    num_query_spans = 0
    with open(output_path, encoding='utf-8', mode='w') as f:
        for sentence in sentences:
            for (token_num, input_token) in enumerate(sentence):
                unannotated_token = input_token.lstrip(
                    QUERY_SPAN_START + CANDIDATE_SPAN_START
                ).rstrip(
                    QUERY_SPAN_END + CANDIDATE_SPAN_END
                )
                unannotated_token_pos = input_token.index(unannotated_token)
                input_token_start = input_token[:unannotated_token_pos]
                input_token_end = input_token[unannotated_token_pos + len(unannotated_token):]
                if unannotated_token.lower() in query_span_words:
                    output_token = CANDIDATE_SPAN_START + unannotated_token + CANDIDATE_SPAN_END
                    output_token = QUERY_SPAN_START + output_token + QUERY_SPAN_END
                    output_token = input_token_start + output_token + input_token_end
                    num_query_spans += 1
                else:
                    output_token = input_token
                if token_num > 0:
                    f.write(' ')
                f.write(output_token)
            f.write('\n')

    logging.info(f'Wrote {num_query_spans} query spans')


def main():
    from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
    parser = ArgumentParser(
        description='Add query span annotations for make-toy-data.py to text file.',
        formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument('input_path',
                        help='Path to input text file, one sentence per line, tokenized with '
                             'tokens separated by spaces, potentially including span annotations '
                             'in the same format as the output file.')
    parser.add_argument('output_path',
                        help='Path to output file with query spans in [square brackets] and '
                             'candidate spans in <angle brackets>.')
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s %(name)-12s %(levelname)-8s %(message)s')

    add_query_spans_to_toy_data_input(args.input_path, args.output_path)


if __name__ == '__main__':
    main()

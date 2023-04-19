#!/usr/bin/env python3

import logging

from coref import read_jsonl, write_jsonl


def make_hit_span(sentence_index, span):
    return dict(
        sentenceIndex=sentence_index,
        startToken=span[0],
        endToken=span[1],
    )


def convert_docs(docs, max_context_size=5, min_context_size=None):
    if min_context_size is None:
        min_context_size = max_context_size
    elif min_context_size > max_context_size:
        raise Exception('min context size is {} but max context size is {}'.format(
            min_context_size, max_context_size))

    hit_max_num_sentences = max_context_size + 1
    hit_min_num_sentences = min_context_size + 1

    def _iter():
        total_num_sentences = 0
        total_num_hits = 0
        for doc in docs:
            total_num_sentences += len(doc['sentences'])
            for end_sentence_index in range(hit_min_num_sentences, len(doc['sentences'])):
                start_sentence_index = max(0, end_sentence_index - hit_max_num_sentences)
                sentences = doc['sentences'][start_sentence_index:end_sentence_index]
                query_spans = [
                    make_hit_span(len(sentences) - 1, span)
                    for span in sentences[-1]['query_spans']
                ]
                candidate_spans = [
                    make_hit_span(i, span)
                    for (i, s) in enumerate(sentences)
                    for span in s['candidate_spans']
                ]
                if query_spans:
                    yield dict(
                        documentId=doc['document_id'],
                        startSentenceIndex=start_sentence_index,
                        endSentenceIndex=end_sentence_index,
                        sentenceIds=[s['sentence_id'] for s in sentences],
                        sentences=[s['tokens'] for s in sentences],
                        querySpans=query_spans,
                        candidateSpans=candidate_spans,
                    )
                    total_num_hits += 1

        logging.info('{} sentences processed, {} HITs output'.format(
            total_num_sentences, total_num_hits))

    return _iter()


def main():
    from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
    parser = ArgumentParser(
        description='Convert spans extracted from UDS into HIT-ready format.',
        formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument('input_path', help='Path to input jsonl file.')
    parser.add_argument('output_path', help='Path to output jsonl file.')
    parser.add_argument('--max-context-size', type=int, default=5,
                        help='Maximum number of sentences to show before query sentence.')
    parser.add_argument('--min-context-size', type=int,
                        help='Minimum number of sentences to show before query sentence '
                             ' (default: same as max).')
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s %(name)-12s %(levelname)-8s %(message)s')

    write_jsonl(
        convert_docs(
            read_jsonl(args.input_path),
            max_context_size=args.max_context_size,
            min_context_size=args.min_context_size),
        args.output_path)


if __name__ == '__main__':
    main()

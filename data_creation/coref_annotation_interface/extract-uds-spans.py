#!/usr/bin/env python3

from collections import defaultdict, deque
import logging
import os

from decomp import UDSCorpus

from coref import SyntaxTreeNode, get_semantics_span, write_jsonl, iter_sentences_as_docs


LOGGER = logging.getLogger(__name__)

QUERY_3PP_SPAN_WORDS = {
    ('they',),
    ('them',),
    ('themself',),
    ('themselves',),
    ('their',),
    ('theirs',),
    ('she',),
    ('her',),
    ('herself',),
    ('hers',),
    ('he',),
    ('him',),
    ('himself',),
    ('his',),
}

QUERY_TAGS = {
    'PRP',
    'PRP$',
}

CANDIDATE_TAGS_LIMITED = {
    'NNP',
    'NNPS',
    'NP',
    'NML',
    'PRP',
    'PRP$',
    'WP',
    'WP$',
}

CANDIDATE_TAGS_ALL = CANDIDATE_TAGS_LIMITED.union({
    'NN',
    'NNS',
})

QUERY_ALL = 'all'
QUERY_3PP = '3pp'

SYNTAX_ALL = 'all'
SYNTAX_LIMITED = 'limited'

SEMANTICS_ALL = 'all'
SEMANTICS_WORDSENSE = 'wordsense'


def load_doc_parse_strs(parse_root):
    parse_strs = defaultdict(deque)
    for domain in os.listdir(parse_root):
        parse_parent = os.path.join(parse_root, domain, 'penntree')
        if os.path.isdir(parse_parent):
            for filename in os.listdir(parse_parent):
                if filename.endswith('.xml.tree'):
                    document_id = domain + '-' + filename[:-len('.xml.tree')]
                    parse_path = os.path.join(parse_parent, filename)
                    with open(parse_path, encoding='utf-8') as f:
                        for line in f:
                            parse_strs[document_id].append(line.strip())

    return parse_strs


def _get_syntax_spans(tree, tags, include_np_prefix=False):
    return [
        n.get_span() for n in tree.get_nodes()
        if n.tag is not None and (
            n.tag in tags or (include_np_prefix and n.tag.startswith('NP'))
        )
    ]


def get_span_words(span, tokens):
    return tuple(w.lower() for w in tokens[span[0]:span[1]])


def get_query_syntax_spans(tree, tokens, span_types=QUERY_ALL):
    return [
        span
        for span in _get_syntax_spans(tree, QUERY_TAGS, include_np_prefix=False)
        if span_types == QUERY_ALL or (
            span_types == QUERY_3PP and get_span_words(span, tokens) in QUERY_3PP_SPAN_WORDS
        )
    ]


def get_candidate_syntax_spans(tree, span_types=SYNTAX_ALL):
    return _get_syntax_spans(
        tree,
        CANDIDATE_TAGS_ALL if span_types == SYNTAX_ALL else CANDIDATE_TAGS_LIMITED,
        include_np_prefix=True)


def get_semantics_spans(uds_graph, span_types=SEMANTICS_ALL):
    return [
        span
        for span in [
            get_semantics_span(uds_graph, node_name)
            for (node_name, node) in uds_graph.semantics_nodes.items()
            if (
                span_types == SEMANTICS_ALL or
                span_types == SEMANTICS_WORDSENSE and 'wordsense' in node
            )
        ]
        if span is not None
    ]


def copy_leaf_text(good_tree, bad_tree):
    good_leaves = good_tree.get_sorted_leaves()
    bad_leaves = bad_tree.get_sorted_leaves()
    if len(good_leaves) != len(bad_leaves):
        raise Exception('no. leaves of trees do not match')
    for (good_leaf, bad_leaf) in zip(good_leaves, bad_leaves):
        if bad_leaf.text != good_leaf.text:
            LOGGER.warning('Changing leaf text: {} -> {}'.format(bad_leaf.text, good_leaf.text))
            bad_leaf.text = good_leaf.text


def check_ptb_tree(ptb_tree, uds_tree, fix_leaf_text=False):
    ptb_tokens = ptb_tree.get_tokens()
    uds_tokens = uds_tree.get_tokens()
    if ptb_tokens != uds_tokens:
        sentences_str = '{}, {}'.format(' '.join(ptb_tokens), ' '.join(uds_tokens))
        if len(ptb_tokens) != len(uds_tokens):
            raise Exception('PTB, UDS sentences differ in length: ' + sentences_str)
        else:
            LOGGER.warning('PTB, UDS sentences differ: ' + sentences_str)
            if fix_leaf_text:
                copy_leaf_text(uds_tree, ptb_tree)


def extract_uds_spans(output_path, split=None, query_span_types=None,
                      candidate_syntax_span_types=None, candidate_semantics_span_types=None,
                      parse_root=None, fix_leaf_text=False):
    def _iter():
        corpus = UDSCorpus(split=split)
        doc_parse_strs = (load_doc_parse_strs(parse_root) if parse_root is not None else {})

        for (i, (_, uds_graph)) in enumerate(corpus.items()):
            if (i + 1) % 1000 == 0:
                LOGGER.info('Extracting spans from graph {}/{}'.format(i + 1, len(corpus)))

            tree = SyntaxTreeNode.from_uds_graph(uds_graph)
            tokens = tree.get_tokens()
            if parse_root is not None:
                ptb_tree = SyntaxTreeNode.from_ptb_str(
                    doc_parse_strs[uds_graph.document_id].popleft())
                check_ptb_tree(ptb_tree, tree, fix_leaf_text=fix_leaf_text)
                tree = ptb_tree

            sentence = dict(
                document_id=uds_graph.document_id,
                sentence_id=uds_graph.sentence_id,
                tokens=tokens,
                query_spans=sorted(list(set(
                    get_query_syntax_spans(tree, tokens, span_types=query_span_types)))),
                candidate_spans=sorted(list(set(
                    (get_semantics_spans(uds_graph, span_types=candidate_semantics_span_types)
                        if candidate_semantics_span_types is not None
                        else []) +
                    (get_candidate_syntax_spans(tree, span_types=candidate_syntax_span_types)
                        if candidate_syntax_span_types is not None
                        else [])
                ))),
            )
            yield sentence

    LOGGER.info('Input corpus split: {}'.format(split if split is not None else '(all)'))
    LOGGER.info('Query spans: {}'.format(query_span_types))
    LOGGER.info('Candidate syntax spans: {}'.format(
        candidate_syntax_span_types if candidate_syntax_span_types is not None else '(none)'))
    LOGGER.info('Candidate semantics spans: {}'.format(
        candidate_semantics_span_types
        if candidate_semantics_span_types is not None
        else '(none)'))
    LOGGER.info('PTB parse root dir: {}'.format(
        parse_root if parse_root is not None else '(none)'))
    LOGGER.info('Fix leaf text: {}'.format(fix_leaf_text))
    LOGGER.info('Output path: {}'.format(output_path))

    LOGGER.info('Extracting spans ...')
    write_jsonl(iter_sentences_as_docs(_iter()), output_path)


def main():
    from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
    parser = ArgumentParser(
        description='Compute spans in UDS corpus and write to jsonl file, one document per line.',
        formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument('output_path', help='Path to output jsonl file.')
    parser.add_argument('--split', choices=('train', 'dev', 'test'),
                        help='UDS split to use (default: all splits)')
    parser.add_argument('--query-spans', choices=(QUERY_ALL, QUERY_3PP), default=QUERY_ALL,
                        help='Include all query spans or just third-person personal pronouns')
    parser.add_argument('--candidate-syntax-spans', choices=(SYNTAX_ALL, SYNTAX_LIMITED),
                        help='Include candidate spans from syntax graph '
                             '(spans corresponding to all noun tags or limited set of tags)')
    parser.add_argument('--candidate-semantics-spans',
                        choices=(SEMANTICS_ALL, SEMANTICS_WORDSENSE),
                        help='Include candidate spans from semantics graph '
                             '(all spans or only those with wordsense properties)')
    parser.add_argument('--parse-root', help='Path to root directory of PTB parses for corpus')
    parser.add_argument('--fix-leaf-text', action='store_true',
                        help='Fix PTB leaf text with text from UDS')
    args = parser.parse_args()

    LOGGER.setLevel(logging.INFO)
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter('%(asctime)-15s %(levelname)-8s %(message)s'))
    LOGGER.addHandler(handler)

    extract_uds_spans(args.output_path, split=args.split,
                      query_span_types=args.query_spans,
                      candidate_syntax_span_types=args.candidate_syntax_spans,
                      candidate_semantics_span_types=args.candidate_semantics_spans,
                      parse_root=args.parse_root,
                      fix_leaf_text=args.fix_leaf_text)


if __name__ == '__main__':
    main()

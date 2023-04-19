#!/usr/bin/env python3

from coref import iter_jsonl, human_format_cluster


def main():
    from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
    parser = ArgumentParser(
        description='Explore coref data in json lines format.',
        formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument('input_path', help='Path to coref data in json lines format')
    args = parser.parse_args()
    for j in iter_jsonl(args.input_path):
        words = [w for s in j['sentences'] for w in s]
        text = ' '.join(words)
        print()
        print(j['doc_key'])
        print(text)
        for cluster in j['clusters']:
            print(human_format_cluster([(s[0], s[1] + 1) for s in cluster], words))


if __name__ == '__main__':
    main()

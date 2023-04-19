import json
import sys
import os

# f = open(sys.argv[1], 'r')
# all_lines = [json.loads(l) for l in f]
#
# ments = 0
# clus = 0
# utt = 0
#
# for line in all_lines:
#     clus += len(line["clusters"])
#     all_ments = [a for b in line["clusters"] for a in b ]
#     ments += len(all_ments)
#     utt += max(line["sentence_map"])
#     utt += 1
#
#
# print(f"Utter: {utt}, Ment: {ments}, Clus: {clus}")
# print(f"{utt} & {ments} & {clus}")


for file_name in os.listdir(sys.argv[1]):
    f = open(os.path.join(sys.argv[1], file_name), 'r')

    all_lines = [json.loads(l) for l in f]

    ments = 0
    clus = 0
    utt = 0
    words = 0
    num_scene = 0
    for line in all_lines:
        # if line['doc_key'][9] == "t":
        #     # print(line['doc_key'])
        #     continue
        # print(line['doc_key'], line['doc_key'][9])
        # print(line)
        num_scene += 1
        clus += len(line["clusters"])
        words += len(line['tokens'])
        all_ments = [a for b in line["clusters"] for a in b]
        ments += len(all_ments)
        try:
            utt += max(line["sentence_map"])
        except:
            print(line['doc_key'])

    print(os.path.join(sys.argv[1], file_name))
    print(f"Scene: {num_scene}, Utter: {utt}, Ment: {ments}, Clus: {clus}, Words: {words}")
    # print(f"{utt} & {ments} & {clus}")

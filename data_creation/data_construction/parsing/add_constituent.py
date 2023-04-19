import pickle as pkl
from copy import deepcopy

import pickle as pkl
import spacy
import csv
import json
from copy import deepcopy
import spacy_stanza
import benepar
from tqdm import tqdm
from copy import deepcopy

import stanza

import benepar, spacy

spacy.prefer_gpu()

berkeley_parser = spacy.load('en_core_web_md')
berkeley_parser.add_pipe('benepar', config={'model': 'benepar_en3_large'})

# file_name = "parsed_corpus_friends_new"
# file_name = "parsed_corpus_friends"
file_name = "parsed_corpus_tbbt"
root = "/brtx/605-nvme2/bzheng12/multi_coref/data_construction/parsing/"
print("File Path:", file_name)


with open(root + file_name + '.pkl', 'rb') as f:
    data = pkl.load(f)

for epi_id in tqdm(data):
    # if epi_id != (1, 1):
    #     continue
    for scene in data[epi_id]:
        for utt in scene:
            if utt['en_subtitles'] != "":
                utterance = utt['en_subtitles']
            else:
                utterance = utt['utterance']

            doc = berkeley_parser(utterance)
            temp = []
            for sent in list(doc.sents):
                for token in sent._.constituents:
                    if len(token._.labels) == 0:
                        continue
                    temp.append((token.text, token.start, token.end, token._.labels[0]))
            utt['berkeley_constituency'] = temp


with open(root + file_name + '_constituent_berkeley_large.pkl', 'wb') as f:
    pkl.dump(data, f)

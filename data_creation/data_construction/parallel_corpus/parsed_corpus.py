import pickle as pkl
import spacy
import csv
import json
from copy import deepcopy
import spacy_stanza
import benepar
from tqdm import tqdm
from copy import deepcopy


with open('parallel_data/tbbt_en_zh.pkl', 'rb') as f:
    corpus = pkl.load(f)


sm_parser = spacy.load('en_core_web_sm')
berkeley_parser = spacy.load('en_core_web_md')
berkeley_parser.add_pipe("benepar", config={"model": "benepar_en3"})
# stanza_parser = spacy_stanza.load_pipeline('en')
trf_parser = spacy.load("en_core_web_trf")


# output = {}
# # Iterate each episode in the corpus
# for epi_id in tqdm(list(corpus.keys())):
#     scenes = deepcopy(corpus[epi_id])
#     # Process each scene
#     # Parse utterance in each scene and add the parsed results to the utterance
#     for scene in scenes:
#         for instance in scene:
#             if 'en_subtitles' not in instance:
#                 continue
#             en_subtitles = [x.strip().lstrip('-').lstrip().lstrip('.').lstrip() for x in instance['en_subtitles']]
#             instance['en_subtitles'] = en_subtitles
#
#             text = " ".join(en_subtitles)
#             utterance = sm_parser(text)
#             instance['sm_noun_chunk'] = [(item.text, item.start, item.end) for item in utterance.noun_chunks]
#             instance['sm_pron'] = [(item.text,i, i+1) for i, item in enumerate(utterance) if item.pos_=="PRON"]
#
#             utterance = berkeley_parser(text)
#             instance['berkeley_noun_chunk'] = [(item.text, item.start, item.end) for item in utterance.noun_chunks]
#             instance['berkeley_pron'] = [(item.text,i, i+1) for i, item in enumerate(utterance) if item.pos_=="PRON"]
#
#             # utterance = stanza_parser(text)
#             # instance['stanza_noun_chunk'] = [(item.text, item.start, item.end) for item in utterance.noun_chunks]
#             # instance['stanza_pron'] = [(item.text,i, i+1) for i, item in enumerate(utterance) if item.pos_=="PRON"]
#
#             utterance = trf_parser(text)
#             instance['trf_noun_chunk'] = [(item.text, item.start, item.end) for item in utterance.noun_chunks]
#             instance['trf_pron'] = [(item.text,i, i+1) for i, item in enumerate(utterance) if item.pos_=="PRON"]
#     output[epi_id] = scenes

output = {}
# Iterate each episode in the corpus
for epi_id in tqdm(list(corpus.keys())):
    scenes = deepcopy(corpus[epi_id])
    # Process each scene
    # Parse utterance in each scene and add the parsed results to the utterance
    for scene in scenes:
        for instance in scene:
            text = instance['sentence']
            utterance = sm_parser(text)
            instance['sm_noun_chunk'] = [(item.text, item.start, item.end) for item in utterance.noun_chunks]
            instance['sm_pron'] = [(item.text,i, i+1, item.pos_, item.tag_) for i, item in enumerate(utterance)]

            utterance = berkeley_parser(text)
            instance['berkeley_noun_chunk'] = [(item.text, item.start, item.end) for item in utterance.noun_chunks]
            instance['berkeley_pron'] = [(item.text,i, i+1, item.pos_, item.tag_) for i, item in enumerate(utterance)]

            # utterance = stanza_parser(text)
            # instance['stanza_noun_chunk'] = [(item.text, item.start, item.end) for item in utterance.noun_chunks]
            # instance['stanza_pron'] = [(item.text,i, i+1) for i, item in enumerate(utterance) if item.pos_=="PRON"]

            utterance = trf_parser(text)
            instance['trf_noun_chunk'] = [(item.text, item.start, item.end) for item in utterance.noun_chunks]
            instance['trf_pron'] = [(item.text,i, i+1, item.pos_, item.tag_) for i, item in enumerate(utterance)]
    output[epi_id] = scenes

with open('parallel_data/parsed_corpus_all_zh.pkl', 'wb') as f:
    pkl.dump(output, f)










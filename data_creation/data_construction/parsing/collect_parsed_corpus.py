import pickle as pkl
import spacy
import csv
import json
from copy import deepcopy
import spacy_stanza
import benepar
from tqdm import tqdm
from copy import deepcopy

with open('friends_three_way_new.pkl', 'rb') as f:
    corpus = pkl.load(f)

sm_parser = spacy.load('en_core_web_sm')
berkeley_parser = spacy.load('en_core_web_md')
berkeley_parser.add_pipe("benepar", config={"model": "benepar_en3"})
trf_parser = spacy.load("en_core_web_trf")

output = {}
# Iterate each episode in the corpus
for epi_id in tqdm(list(corpus.keys())):
    scenes = deepcopy(corpus[epi_id])
    # Process each scene
    # Parse utterance in each scene and add the parsed results to the utterance
    for scene in scenes:
        for instance in scene:
            if instance['en_subtitles'] != "":
                text = instance['en_subtitles']
            else:
                text = instance['utterance']

            utterance = sm_parser(text)
            instance['sm_noun_chunk'] = [(item.text, item.start, item.end) for item in utterance.noun_chunks]
            instance['sm_pron'] = [(item.text, i, i + 1, item.pos_, item.tag_) for i, item in enumerate(utterance)]

            utterance = berkeley_parser(text)
            instance['berkeley_noun_chunk'] = [(item.text, item.start, item.end) for item in utterance.noun_chunks]
            instance['berkeley_pron'] = [(item.text, i, i + 1, item.pos_, item.tag_) for i, item in
                                         enumerate(utterance)]

            # utterance = stanza_parser(text)
            # instance['stanza_noun_chunk'] = [(item.text, item.start, item.end) for item in utterance.noun_chunks]
            # instance['stanza_pron'] = [(item.text,i, i+1) for i, item in enumerate(utterance) if item.pos_=="PRON"]

            utterance = trf_parser(text)
            instance['trf_noun_chunk'] = [(item.text, item.start, item.end) for item in utterance.noun_chunks]
            instance['trf_pron'] = [(item.text, i, i + 1, item.pos_, item.tag_) for i, item in enumerate(utterance)]
    output[epi_id] = scenes

with open('parsed_corpus_friends_new.pkl', 'wb') as f:
    pkl.dump(output, f)

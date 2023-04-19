import pickle as pkl
import spacy
import csv
import json
from copy import deepcopy
from tqdm import tqdm
import benepar
import jiwer
import re
import xlsxwriter
from string import punctuation
from data_construction.alignment.fine_alignment.utils.preprocessing import fetch_subsets
from data_construction.alignment.fine_alignment.utils.preprocessing import fetch_subsets_a4k
from data_construction.alignment.fine_alignment.utils.preprocessing import organize_coarse_alignment_by_seasons
from data_construction.alignment.fine_alignment.utils.helper_functions import *

# Define sentence transformation
transformation = jiwer.Compose([
    jiwer.ToLowerCase(),
    jiwer.RemoveMultipleSpaces(),
    jiwer.ExpandCommonEnglishContractions(),
    jiwer.RemovePunctuation(),
    jiwer.Strip()
])


def count_scene_mentions(data):
    temp = []
    for scene in data:
        temp.append(len(scene['querySpans']))
    return temp


def extract_scenes(parsed_corpus, transcript_label):
    sm_parser = spacy.load('en_core_web_sm')
    output = []
    for epi_key in tqdm(parsed_corpus.keys()):
        for i in range(len(parsed_corpus[epi_key])):
            scene_id = "s" + str(epi_key[0]).zfill(2) + "e" + str(epi_key[1]).zfill(2) + "c" + str(i).zfill(
                2) + transcript_label
            all_sentences = []
            all_query_spans = []

            all_utterances = []
            all_utterance_with_infos = []
            all_en_subtitles = []
            all_zh_subtitles = []
            all_fa_subtitles = []

            scene_tag = parsed_corpus[epi_key][i]
            j = 0
            for utt in scene_tag:
                all_utterances.append(utt['utterance'])
                all_utterance_with_infos.append(utt['utterance_with_info'])
                all_en_subtitles.append(utt['en_subtitles'])
                all_zh_subtitles.append(utt['zh_subtitles'])
                all_fa_subtitles.append(utt['fa_subtitles'])

                speaker = utt['speaker'].strip().strip("(").strip(")").strip().strip(".").strip().strip(":")
                speaker_tokens = [item.text for item in sm_parser(speaker)]
                if utt["en_subtitles"] != "":
                    sentence = utt['en_subtitles']
                else:
                    sentence = utt['utterance']
                sentence_token = [item[0] for item in utt['sm_pron']]
                sm_nps = process_nps_punctuation(sentence_token,
                                                 process_nps_punctuation(sentence_token, utt['sm_noun_chunk']))
                berkeley_nps = process_nps_punctuation(sentence_token, process_nps_punctuation(sentence_token, utt[
                    'berkeley_noun_chunk']))
                trf_nps = process_nps_punctuation(sentence_token,
                                                  process_nps_punctuation(sentence_token, utt['trf_noun_chunk']))
                noun_phrase = merge_maximum_span(list(set(sm_nps) | set(berkeley_nps) | set(trf_nps)))
                temp_pron = []
                temp_pron.extend([(item[0], item[1], item[2]) for item in utt['sm_pron'] if item[3] == 'PRON'])
                temp_pron.extend([(item[0], item[1], item[2]) for item in utt['berkeley_pron'] if item[3] == 'PRON'])
                temp_pron.extend([(item[0], item[1], item[2]) for item in utt['trf_pron'] if item[3] == 'PRON'])
                pron = merge_maximum_span(list(set(temp_pron)))
                for item in utt['berkeley_noun_chunk']:
                    temp = list(deepcopy(item))
                    if sentence_token[temp[1]] in punctuation:
                        # If the first token is quotation and there is not only one quotation, stop removing
                        if sentence_token[temp[1]] == "\"":
                            if " ".join(sentence_token[temp[1]: temp[2]]).count("\"") != 1:
                                continue
                        temp[1] -= 1
                    elif " ".join(sentence_token[temp[1]: temp[2]]).count("\"") % 2 == 1:
                        if sentence_token[temp[2]] == "\"":
                            pass
                mention = list(set(noun_phrase) | set(pron))
                mention.sort(key=lambda x: x[1])

                all_sentences.append(speaker_tokens + [":"] + sentence_token)
                for span in mention:
                    all_query_spans.append({
                        "sentenceIndex": j,
                        "startToken": span[1] + len(speaker_tokens) + 1,
                        "endToken": span[2] + len(speaker_tokens) + 1
                    })
                j += 1
            output.append({
                "sentences": all_sentences,
                "querySpans": all_query_spans,
                "candidateSpans": all_query_spans,
                "clickSpans": all_query_spans,
                "utterances": all_utterances,
                "utterances_with_info": all_utterance_with_infos,
                "en_subtitles": all_en_subtitles,
                "zh_subtitles": all_zh_subtitles,
                "fa_subtitles": all_fa_subtitles,
                "scene_id": scene_id
            })
    return output


def combine_samples(instances):
    # Store Data
    sentences = []
    querySpans = []
    candidateSpans = []
    clickSpans = []
    sentence_offsets = [0]
    querySpans_offsets = [0]

    for instance in instances:
        offset = len(sentences)
        sentences.extend(instance['sentences'])
        for item in instance['querySpans']:
            token = deepcopy(item)
            token['sentenceIndex'] = item['sentenceIndex'] + offset
            querySpans.append(token)
        for item in instance['candidateSpans']:
            token = deepcopy(item)
            token['sentenceIndex'] = item['sentenceIndex'] + offset
            candidateSpans.append(token)
        for item in instance['clickSpans']:
            token = deepcopy(item)
            token['sentenceIndex'] = item['sentenceIndex'] + offset
            clickSpans.append(token)
        sentence_offsets.append(len(sentences))
        querySpans_offsets.append(len(querySpans))
    return {
        "sentences": sentences,
        "querySpans": querySpans,
        "candidateSpans": candidateSpans,
        "clickSpans": clickSpans,
        "sentence_offsets": sentence_offsets,
        "querySpans_offsets": querySpans_offsets,
    }


# Merge
def merge_maximum_span(spans):
    spans.sort(key=lambda x: x[1])
    # print(spans)
    to_pop = []
    for j, (word_0, start_idx_0, end_idx_0) in enumerate(spans):
        for k, (word_1, start_idx_1, end_idx_1) in enumerate(spans):
            if k == j:
                continue
            if (start_idx_1 >= start_idx_0) and (end_idx_1 <= end_idx_0):
                to_pop.append(spans[k])
    for item in to_pop:
        if item in spans:
            spans.remove(item)

    spans.sort(key=lambda x: x[1])

    return spans


def clean_sentence_brackets(text):
    text = text.strip().lstrip('-').lstrip().lstrip('.').lstrip()
    new_text = re.sub(u"\\(.*?\\)|\\{.*?}|\\[.*?]", "", text)
    return new_text


def clean_sentences(scenes):
    output = []
    for scene in scenes:
        temp = []
        for item in scene:
            if "en_subtitles" in item:
                temp_sents = " ".join(item['en_subtitles'])
            else:
                temp_sents = item['utterance']
            item["sentence_with_info"] = temp_sents
            cleaned_sentence = clean_sentence_brackets(temp_sents)
            item["sentence"] = cleaned_sentence
            temp.append(item)
        output.append(temp)
    return output


def process_span_punctuation(sentence_tokens, span):
    temp = list(deepcopy(span))
    if sentence_tokens[temp[1]] in punctuation:
        # If the first token is quotation and there is not only one quotation, stop removing
        if sentence_tokens[temp[1]] == "\"":
            if " ".join(sentence_tokens[temp[1]: temp[2]]).count("\"") != 1:
                return temp
        temp[1] += 1
    elif " ".join(sentence_tokens[temp[1]: temp[2]]).count("\"") % 2 == 1:
        if sentence_tokens[temp[2]] == "\"":
            temp[2] += 1
    return tuple(temp)


def process_nps_punctuation(sentence_tokens, nps):
    output = []
    for np in nps:
        temp_np = process_span_punctuation(sentence_tokens, np)
        if temp_np[1] != temp_np[2]:
            output.append(tuple(temp_np))
    return output


def is_chinese(token):
    for x in token:
        if u'\u4e00' <= x <= u'\u9fff':
            return True
    return False

import pickle as pkl
import spacy
import csv
import json
from copy import deepcopy
from tqdm import tqdm
import benepar
import jiwer
import re
from string import punctuation
from data_construction.alignment.fine_alignment.utils.preprocessing import fetch_subsets
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


class ParallelCorpusCollector:
    def __init__(self,
                 en_subtitle_path,
                 other_subtitle_path,
                 transcript_path,
                 coarse_alignment_path,
                 fine_alignment_path,
                 root_path,
                 output_path):
        # Config Path
        self.en_subtitle_path = en_subtitle_path
        self.other_subtitle_path = other_subtitle_path
        self.transcript_path = transcript_path
        self.coarse_alignment_path = coarse_alignment_path
        self.fine_alignment_path = fine_alignment_path
        self.root_path = root_path
        self.output_path = output_path

        with open(self.en_subtitle_path, 'rb') as f:
            self.en_subtitle = pkl.load(f)
        with open(self.other_subtitle_path, 'rb') as f:
            self.zh_subtitle = pkl.load(f)
        with open(self.transcript_path, 'rb') as f:
            self.transcripts = pkl.load(f)
        with open(self.coarse_alignment_path, 'rb') as f:
            temp = pkl.load(f)
        self.results = organize_coarse_alignment_by_seasons(temp)
        with open(self.fine_alignment_path, 'rb') as f:
            self.all_alignment = pkl.load(f)

    def collect_parallel_episode(self, season_id, episode_id):
        """
        Collect parallel in (season_id, episode_id) to build parallel corpus
        """
        # Fetch subset data
        (en_subset, zh_subset, tbbt_episode) = fetch_subsets(
            episode=self.transcripts,
            en_subtitle=self.en_subtitle,
            zh_subtitle=self.zh_subtitle,
            results=self.results,
            season_id=season_id,
            episode_id=episode_id,
            bias=200,
            zh_split=True
        )

        # Construct the index dictionary from original index to the collected index
        idx_dict = {}
        idx = 0
        for i, x in enumerate(tbbt_episode):
            while True:
                if x[0] == self.transcripts[(season_id, episode_id)][idx][0] and x[1] == \
                        self.transcripts[(season_id, episode_id)][idx][1]:
                    idx_dict[idx] = i
                    idx += 1
                    break
                else:
                    idx += 1

        # Collect ZH subtitles to episodes
        alignment = self.all_alignment[(season_id, episode_id)]

        one_episode = []
        # Turn episode into a dictionary form
        for x in tbbt_episode:
            temp = {'speaker': x[1], 'utterance': clean_sentence_brackets(x[0]), 'utterance_with_info': x[0],
                    'en_subtitles': "", 'zh_subtitles': ""}
            one_episode.append(temp)

        # Add subtitles into episode
        for x in alignment:
            en_subs = []
            zh_subs = []
            for item in alignment[x]:
                en_subs.append(en_subset[item])
                zh_subs.append(zh_subset[item])
            one_episode[x]['en_subtitles'] = " ".join(en_subs)
            one_episode[x]['zh_subtitles'] = " ".join(zh_subs)

        return one_episode

    def show_episode_alignment_result(self, season_id, episode_id):
        """
        Show the percentage of aligned episode and subtitles
        """
        item = (season_id, episode_id)
        epi2sub = self.all_alignment[item]
        sub2epi = turn_sub2epi_into_epi2sub(self.all_alignment[item])

        num_episode_all = len(self.transcripts[item])
        num_episode_aligned = len(epi2sub)

        num_subtitle_all = max(sub2epi) - min(sub2epi)
        num_subtitle_aligned = len(sub2epi)

        output = (
            num_episode_all,
            num_episode_aligned,
            num_subtitle_all,
            num_subtitle_aligned
        )

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
            output.append(temp_np)
    return output

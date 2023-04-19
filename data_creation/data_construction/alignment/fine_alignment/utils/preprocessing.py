import pickle as pkl
import json
from collections import defaultdict
import jiwer
from copy import deepcopy
import re
import string

# Set the data cleaning method
transformation = jiwer.Compose([
    jiwer.ToLowerCase(),
    jiwer.RemoveMultipleSpaces(),
    jiwer.ExpandCommonEnglishContractions(),
    jiwer.RemovePunctuation(),
    jiwer.Strip()
])


def organize_coarse_alignment_by_seasons(all_data):
    """
    Organize the indexs of alignment result by season and episode
    Input: all alignment index
    Ouput: Dictionary of alignment index
            {season_id: {
                episode_id: []
            }
            }
    """
    res = {}
    for epi in list(all_data.keys()):
        season = int(epi[1:3])
        episode = int(epi[-2:])
        # Process the bck in one episode
        temp = get_index_dict(all_data[epi])
        if season not in res:
            res[season] = {
                episode: temp
            }
        else:
            res[season][episode] = temp
    return res


def get_index_dict(episode):
    """
    Organize index within one episode
    Input: ([[]])
    Output: {index: []}
    """
    index_dict = defaultdict()
    for idx, segment, en_sub, zh_sub in episode:
        temp = [segment, en_sub, zh_sub]
        if idx not in index_dict:
            index_dict[idx] = [temp]
        else:
            index_dict[idx].append(temp)
    return index_dict


def clean_sentence_brackets(text):
    text = text.strip().lstrip('-').lstrip().lstrip('.').lstrip("－").lstrip()
    new_text = re.sub(u"\\(.*?\\)|\\{.*?}|\\[.*?]", "", text)
    return new_text


def find_all_continuous_subsets(idx_list, gaps, len_threshold, gap_threshold):
    """
    Locate continuous subset that gap between to indexs is small than threshold
    Input: indexs, gaps, threshold
    Output: indexs of continuous subset
    """
    res = []
    path = [idx_list[0]]
    for i in range(len(gaps)):
        if gaps[i] <= gap_threshold:
            path.append(idx_list[i + 1])
        else:
            if len(path) >= len_threshold:
                res.append(path)
            path = [idx_list[i + 1]]
    return res


def calculate_gaps(idx_list):
    """
    Calculate gaps between elements given an list of integer
    """
    gaps = []
    idx_list.sort()
    for i in range(len(idx_list) - 1):
        gaps.append(idx_list[i + 1] - idx_list[i])
    return gaps


def get_epi_indexs_gaps(episode):
    """
    Get all the indexs within one episode
    Input: {index: [[]]}
    Output: sorted index list
    """
    idx_list = []
    for idx in episode:
        idx_list.append(idx)
    idx_list.sort()
    # Calculate gaps
    gaps = calculate_gaps(idx_list)
    return idx_list, gaps


def fetch_subsets(episode, en_subtitle, zh_subtitle, results, season_id, episode_id, bias, zh_split=False):
    """
    episode: Whole transcript
    en_subtitle: all en subtitle
    zh_subtitle: all subtitle in other language
    results: the coarse alignment result
    season_id: id of the season to align
    episode_id: id of episode to align
    bias: [first_index-bias, last_index+bias] is the list of subtitles to align
        a subset of en_subtitle, a subset of subtitles in another language and the corresponding utterance

    Special Notice: Since in Open Subtitle some subtitles contains two utterance split with "-", we split them
    """
    idx_list, gaps = get_epi_indexs_gaps(results[season_id][episode_id])
    subsets = find_all_continuous_subsets(idx_list, gaps, 6, 100)[-1]
    # Calculate gaps within the subset
    gaps_subsets = calculate_gaps(subsets)

    # Prepare Subtitle Subset
    start = subsets[0] - bias
    end = subsets[-1] + bias
    old_en_subset = [clean_sentence_brackets(item) for item in en_subtitle[start: end]]
    old_zh_subset = [clean_sentence_brackets(item) for item in zh_subtitle[start: end]]
    en_subset = []
    zh_subset = []

    # Split mutiple utterance in subtitle
    for i in range(len(old_en_subset)):
        utt_en = clean_sentence_brackets(old_en_subset[i]).replace("... - ...", "...")
        utt_en_token = utt_en.strip().split()
        utt_zh = clean_sentence_brackets(old_zh_subset[i])

        if zh_split:
            # Split concatenated subtitle with "-"
            if "-" in utt_en_token:
                idx = [k for k, item in enumerate(utt_en_token) if item == "-"][-1]
                if utt_en_token[idx - 1][-1] in string.punctuation:
                    if "－" in utt_zh or "-" in utt_zh:
                        en_subs = [" ".join(utt_en_token[:idx]), " ".join(utt_en_token[idx + 1:])]
                        if "－" in utt_zh:
                            zh_subs = utt_zh.strip().lstrip("－").split("－")
                        else:
                            zh_subs = utt_zh.strip().lstrip("-").split("-")

                        for j in range(len(en_subs)):
                            en_subset.append(en_subs[j])
                            zh_subset.append(zh_subs[j])
                    else:
                        former = " ".join(utt_en_token[:idx])
                        latter = " ".join(utt_en_token[idx + 1:])
                        old_en_subset[i] = former
                        old_en_subset[i + 1] = latter + " " + old_en_subset[i + 1]
                        en_subset.append(old_en_subset[i])
                        zh_subset.append(old_zh_subset[i])
            else:
                en_subset.append(old_en_subset[i])
                zh_subset.append(old_zh_subset[i])
        else:
            en_subset.append(old_en_subset[i])
            zh_subset.append(old_zh_subset[i])

    # Remove Characters
    for i in range(len(en_subset)):
        en_tokens = en_subset[i].strip().split()
        for j, item in enumerate(en_tokens):
            if item.isupper() and len(item) >= 3 and item[-1] == ":":
                en_tokens.pop(j)
        en_subset[i] = " ".join(en_tokens)

    tbbt_episode = []
    for x in episode[(season_id, episode_id)]:
        if x[1] != 'Scene':
            tbbt_episode.append(x)

    # Clean the episode bck
    # 1. Remove empty string
    # 2. Remove duplicate stings
    temp_tbbt_episode = []
    abandon_idx = set()
    for i, x in enumerate(tbbt_episode):
        if transformation(x[0]) in [" ", ""]:
            abandon_idx.add(i)
    for length in range(6):
        length += 1
        for i in range(len(tbbt_episode) - length):
            if tbbt_episode[i][0] == tbbt_episode[i + length][0] and tbbt_episode[i][1] == tbbt_episode[i + length][1]:
                abandon_idx.add(i)

    for i, item in enumerate(tbbt_episode):
        if i not in abandon_idx:
            temp_tbbt_episode.append(item)

    return en_subset, zh_subset, temp_tbbt_episode


# def fetch_subsets(episode, en_subtitle, zh_subtitle, results, season_id, episode_id, bias, zh_split=False):
#     """
#     episode: Whole transcript
#     en_subtitle: all en subtitle
#     zh_subtitle: all subtitle in other language
#     results: the coarse alignment result
#     season_id: id of the season to align
#     episode_id: id of episode to align
#     bias: [first_index-bias, last_index+bias] is the list of subtitles to align
#         a subset of en_subtitle, a subset of subtitles in another language and the corresponding utterance
#
#     Special Notice: Since in Open Subtitle some subtitles contains two utterance split with "-", we split them
#     """
#     idx_list, gaps = get_epi_indexs_gaps(results[season_id][episode_id])
#     subsets = find_all_continuous_subsets(idx_list, gaps, 6, 100)[-1]
#     # Calculate gaps within the subset
#     gaps_subsets = calculate_gaps(subsets)
#
#     # Prepare Subtitle Subset
#     start = subsets[0] - bias
#     end = subsets[-1] + bias
#     old_en_subset = [clean_sentence_brackets(item) for item in en_subtitle[start: end]]
#     old_zh_subset = [clean_sentence_brackets(item) for item in zh_subtitle[start: end]]
#     en_subset = []
#     zh_subset = []
#
#     # Split mutiple utterance in subtitle
#     for i in range(len(old_en_subset)):
#         utt_en = clean_sentence_brackets(old_en_subset[i])
#         utt_en_token = utt_en.strip().split()
#         utt_zh = clean_sentence_brackets(old_zh_subset[i])
#
#         if zh_split:
#             # Split concatenated subtitle with "-"
#             if "-" in utt_en_token:
#                 # print(utt_en)
#                 # print(utt_zh)
#                 idx = [k for k, item in enumerate(utt_en_token) if item == "-"][-1]
#                 if utt_en_token[idx - 1][-1] in string.punctuation:
#                     if "－" in utt_zh or "-" in utt_zh:
#                         print(utt_en)
#                         print(utt_zh)
#                         en_subs = [" ".join(utt_en_token[:idx]), " ".join(utt_en_token[idx + 1:])]
#                         if "－" in utt_zh:
#                             zh_subs = utt_zh.strip().split("－")
#                         else:
#                             zh_subs = utt_zh.strip().split("-")
#
#                         for j in range(len(en_subs)):
#                             en_subset.append(en_subs[j])
#                             zh_subset.append(zh_subs[j])
#                     else:
#                         former = " ".join(utt_en_token[:idx])
#                         latter = " ".join(utt_en_token[idx + 1:])
#                         old_en_subset[i] = former
#                         try:
#                             old_en_subset[i + 1] = latter + " " + old_en_subset[i + 1]
#                         except:
#                             # print(len(old_en_subset), i)
#                             # print(old_en_subset[i])
#                             # print(old_zh_subset[i])
#                             pass
#                         en_subset.append(old_en_subset[i])
#                         zh_subset.append(old_zh_subset[i])
#             else:
#                 en_subset.append(old_en_subset[i])
#                 zh_subset.append(old_zh_subset[i])
#         else:
#             en_subset.append(old_en_subset[i])
#             zh_subset.append(old_zh_subset[i])
#
#     # Remove Characters
#     for i in range(len(en_subset)):
#         en_tokens = en_subset[i].strip().split()
#         for j, item in enumerate(en_tokens):
#             if item.isupper() and len(item) >= 3 and item[-1] == ":":
#                 en_tokens.pop(j)
#         en_subset[i] = " ".join(en_tokens)
#
#     tbbt_episode = []
#     for x in episode[(season_id, episode_id)]:
#         if x[1] != 'Scene':
#             tbbt_episode.append(x)
#
#     # Clean the episode bck
#     # 1. Remove empty string
#     # 2. Remove duplicate stings
#     temp_tbbt_episode = []
#     abandon_idx = set()
#     for i, x in enumerate(tbbt_episode):
#         if transformation(x[0]) in [" ", ""]:
#             abandon_idx.add(i)
#     for length in range(6):
#         length += 1
#         for i in range(len(tbbt_episode) - length):
#             if tbbt_episode[i][0] == tbbt_episode[i + length][0] and tbbt_episode[i][1] == tbbt_episode[i + length][1]:
#                 abandon_idx.add(i)
#
#     for i, item in enumerate(tbbt_episode):
#         if i not in abandon_idx:
#             temp_tbbt_episode.append(item)
#
#     return en_subset, zh_subset, temp_tbbt_episode


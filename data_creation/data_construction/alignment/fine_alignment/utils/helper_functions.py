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


def turn_sub2epi_into_epi2sub(sub2epi):
    """
    Convert dictionary between epi2sub and sub2epi
    """
    epi2sub = {}
    for sub_id in sub2epi:
        for epi_id in sub2epi[sub_id]:
            if epi_id not in epi2sub:
                epi2sub[epi_id] = [sub_id]
            else:
                epi2sub[epi_id].append(sub_id)
    return epi2sub


# Perform index filtering on the alignment seeds
def filter_by_idx(sub2epi):
    """
    Filter index based on the index before and after
    """
    paris = []
    for x in sorted(list(sub2epi.keys())):
        for y in sorted(sub2epi[x]):
            paris.append([x, y])

    res = [paris[0]]
    for i in range(1, len(paris) - 1):
        former = res[-1]
        current = paris[i]
        after = paris[i + 1]
        if former[0] <= current[0] <= after[0]:
            if former[1] <= current[1] <= after[1]:
                res.append(current)
    if paris[-1][0] >= res[-1][0]:
        if paris[-1][1] >= res[-1][1]:
            res.append(paris[-1])

    output = {}
    for x in res:
        sub = x[0]
        epi = x[1]
        if sub not in output:
            output[sub] = [epi]
        else:
            output[sub].append(epi)

    return output


def merge_episode_alignment(dict1, dict2):
    """
    Merge two dictionary into a same dictionary and sort the key
    """
    res = deepcopy(dict1)
    for item in dict2:
        if item in res:
            res[item].extend(dict2[item])
        else:
            res[item] = dict2[item]

    output = {}

    for x in sorted(list(res.keys())):
        output[x] = sorted(list(set(res[x])))

    return output


def get_subset_in_gaps(epi2sub):
    """
    Get the subset pairs between the gaps of episodes and subtitles
    """
    epi_ids = []
    for epi_id in epi2sub:
        epi_ids.append(epi_id)

    abandon = []
    subset_pairs = []
    for i in range(len(epi_ids)-1):
        idx = epi_ids[i]
        idx_after = epi_ids[i+1]
        if idx_after-idx>1:
            # Fetch Episode Subset
            epi_id_subset = list(range(idx+1, idx_after))
            # Fetch Subtitle Subset
            sub_id_subset = list(range(max(epi2sub[idx])+1, min(epi2sub[idx_after])))
            if len(sub_id_subset)>0:
                subset_pairs.append([epi_id_subset, sub_id_subset])
            else:
                abandon.extend(epi_id_subset)
    return subset_pairs, abandon















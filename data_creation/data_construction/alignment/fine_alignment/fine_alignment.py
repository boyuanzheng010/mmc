import pickle as pkl
import json
from collections import defaultdict
import jiwer
from copy import deepcopy
from tqdm import tqdm
import xlsxwriter
import re

from utils.preprocessing import organize_coarse_alignment_by_seasons
from utils.preprocessing import fetch_subsets
from utils.alignment_seeds import *
from utils.helper_functions import *
from utils.alignment_extension import *
from utils.ultimate_alignment import *


# Define sentence transformation
transformation = jiwer.Compose([
    jiwer.ToLowerCase(),
    jiwer.RemoveMultipleSpaces(),
    jiwer.ExpandCommonEnglishContractions(),
    jiwer.RemovePunctuation(),
    jiwer.Strip()
])


# Set File Path
en_subtitle_path = "../../source_data/subtitles/en_zh/en_subtitles.pkl"
other_subtitle_path = "../../source_data/subtitles/en_zh/zh_subtitles.pkl"
transcript_path = "../../source_data/transcripts/tbbt/tbbt_transcripts.pkl"
coarse_alignment_path = "../coarse_alignment/results/tbbt_en_zh.pkl"
results_root_path = "results/tbbt_en_zh/"

# Load Data
with open(en_subtitle_path, 'rb') as f:
    all_en_subtitles = pkl.load(f)
with open(other_subtitle_path, 'rb') as f:
    all_other_subtitles = pkl.load(f)
with open(transcript_path, 'rb') as f:
    all_transcripts = pkl.load(f)

with open(coarse_alignment_path, 'rb') as f:
    temp = pkl.load(f)
    coarse_alignments = organize_coarse_alignment_by_seasons(temp)

# Generate alignment seeds as the start stage
results = {}
for i in range(12):
    for j in tqdm(range(30)):
        try:
            (en_subset, zh_subset, tbbt_episode) = fetch_subsets(
                episode=all_transcripts,
                en_subtitle=all_en_subtitles,
                zh_subtitle=all_other_subtitles,
                results=coarse_alignments,
                season_id=i+1,
                episode_id=j+1,
                bias=200
            )
            temp = generate_alignment_seeds(en_subset, tbbt_episode, window_size=6)
            results[(i+1, j+1)] = temp
        except:
            pass
with open(results_root_path + "0_alignment_seeds.pkl", "wb") as f:
    pkl.dump(results, f)


# Perform Alignment Extension
alignment_seeds = deepcopy(results)
further_alignment = {}
for (i, j) in tqdm(alignment_seeds.keys()):
    (en_subset, zh_subset, tbbt_episode) = fetch_subsets(
                episode=all_transcripts,
                en_subtitle=all_en_subtitles,
                zh_subtitle=all_other_subtitles,
                results=coarse_alignments,
                season_id=i,
                episode_id=j,
                bias=200
            )
    epi2sub = filter_by_idx(turn_sub2epi_into_epi2sub(alignment_seeds[(i,j)]))
    # Extend the neighbor
    while True:
        temp_len = len(turn_sub2epi_into_epi2sub(epi2sub))
        epi2sub = filter_by_idx(add_neighbor_subtitles_to_episode(en_subset, epi2sub, tbbt_episode))
        if temp_len==len(turn_sub2epi_into_epi2sub(epi2sub)):
            break

    # Perform a set of extension
    while True:
        temp_len = len(turn_sub2epi_into_epi2sub(epi2sub))

        gaps, _ = get_subset_in_gaps(epi2sub)
        epi2sub = filter_by_idx(add_strict_match_within_gaps(gaps, epi2sub, en_subset, tbbt_episode))

        gaps, _ = get_subset_in_gaps(epi2sub)
        epi2sub = filter_by_idx(add_wer_match_within_gaps(gaps, epi2sub, en_subset, tbbt_episode))

        gaps, _ = get_subset_in_gaps(epi2sub)
        epi2sub = filter_by_idx(add_wer_substring_match_within_gaps(gaps, epi2sub, en_subset, tbbt_episode))

        epi2sub = filter_by_idx(add_neighbor_subtitles_to_episode(en_subset, epi2sub, tbbt_episode))

        if temp_len==len(turn_sub2epi_into_epi2sub(epi2sub)):
            break

    # Perform alignment within gaps
    while True:
        temp_len = len(turn_sub2epi_into_epi2sub(epi2sub))

        gaps, _ = get_subset_in_gaps(epi2sub)
        epi2sub = filter_by_idx(add_strict_match_within_gaps(gaps, epi2sub, en_subset, tbbt_episode))

        gaps, _ = get_subset_in_gaps(epi2sub)
        epi2sub = filter_by_idx(add_wer_match_within_gaps(gaps, epi2sub, en_subset, tbbt_episode))

        gaps, _ = get_subset_in_gaps(epi2sub)
        epi2sub = filter_by_idx(add_wer_substring_match_within_gaps(gaps, epi2sub, en_subset, tbbt_episode))

        epi2sub = filter_by_idx(extend_neighbors_episode_sliding(en_subset, epi2sub, tbbt_episode))

        if temp_len==len(turn_sub2epi_into_epi2sub(epi2sub)):
            break

    # Extend Subtitle ids with its min and max index
    for x in epi2sub:
        epi2sub[x] = [i for i in range(min(epi2sub[x]), max(epi2sub[x])+1)]

    # Perform ultimate alignment
    gaps = get_final_stage_gap_pairs(epi2sub)
    epi2sub = ultimate_alignment(gaps, epi2sub, en_subset, tbbt_episode)

    further_alignment[(i,j)] = epi2sub

with open(results_root_path + "1_alignment_extension.pkl", "wb") as f:
    pkl.dump(further_alignment, f)


for item in further_alignment:
    epi2sub = further_alignment[item]
    sub2epi = turn_sub2epi_into_epi2sub(further_alignment[item])
    print(item, "Subtitle:", len(sub2epi), len(sub2epi)/(max(sub2epi)-min(sub2epi)), "||", "Episode:", len(further_alignment[item]),  len(further_alignment[item])/len(all_transcripts[item])*100, min(further_alignment[item].keys()))














import pickle as pkl
import json
from collections import defaultdict
import jiwer
from copy import deepcopy
from tqdm import tqdm
import xlsxwriter
import re

from data_construction.alignment.fine_alignment.utils.preprocessing import organize_coarse_alignment_by_seasons
from data_construction.alignment.fine_alignment.utils.preprocessing import fetch_subsets
from data_construction.alignment.fine_alignment.utils.alignment_seeds import *
from data_construction.alignment.fine_alignment.utils.alignment_extension import *
from data_construction.alignment.fine_alignment.utils.helper_functions import *
from data_construction.alignment.fine_alignment.utils.ultimate_alignment import *
from data_construction.alignment.fine_alignment.utils.head_tail_alignment import *
from data_construction.alignment.fine_alignment.utils.visulization import generate_xlxs_for_episode

# Define sentence transformation
transformation = jiwer.Compose([
    jiwer.ToLowerCase(),
    jiwer.RemoveMultipleSpaces(),
    jiwer.ExpandCommonEnglishContractions(),
    jiwer.RemovePunctuation(),
    jiwer.Strip()
])


class FineAligner:
    def __init__(self,
                 en_subtitle_path,
                 other_subtitle_path,
                 transcript_path,
                 coarse_alignment_path,
                 root_path
                 ):
        # Initialize Aligner
        self.root_path = root_path
        self.en_subtitle_path = en_subtitle_path
        self.other_subtitle_path = other_subtitle_path
        self.transcript_path = transcript_path
        self.coarse_alignment_path = coarse_alignment_path

        # Load Data
        with open(self.en_subtitle_path, 'rb') as f:
            self.all_en_subtitles = pkl.load(f)
        print("Complete En Subtitle Loading", len(self.all_en_subtitles))
        with open(self.other_subtitle_path, 'rb') as f:
            self.all_other_subtitles = pkl.load(f)
        print("Complete Other Subtitle Loading", len(self.all_other_subtitles))
        with open(self.transcript_path, 'rb') as f:
            self.all_transcripts = pkl.load(f)
        print("Complete All Transcript Loading")
        with open(self.coarse_alignment_path, 'rb') as f:
            temp = pkl.load(f)
            self.coarse_alignments = organize_coarse_alignment_by_seasons(temp)
        print("Complete Coarse Alignments Loading")

    def alignment_seeds_generation(self):
        print("==========Start Alignment Seeds Generation==========")
        results = {}
        for i in sorted(list(self.coarse_alignments.keys())):
            for j in sorted(list(self.coarse_alignments[i].keys())):
                if (i, j) not in self.all_transcripts:
                    continue
                print("Season:", i, "  Episode:", j)
                try:
                    (en_subset, zh_subset, tbbt_episode) = fetch_subsets(
                        episode=self.all_transcripts,
                        en_subtitle=self.all_en_subtitles,
                        zh_subtitle=self.all_other_subtitles,
                        results=self.coarse_alignments,
                        season_id=i,
                        episode_id=j,
                        bias=200,
                        zh_split=True
                    )
                    temp = generate_alignment_seeds(en_subset, tbbt_episode, window_size=12)
                    if temp != {}:
                        results[(i, j)] = temp
                except:
                    pass

        with open(self.root_path + "0_alignment_seeds.pkl", "wb") as f:
            pkl.dump(results, f)
        return results

    def alignment_extension(self, alignment_seeds):
        further_alignment = {}
        for (i, j) in alignment_seeds.keys():
            (en_subset, zh_subset, tbbt_episode) = fetch_subsets(
                episode=self.all_transcripts,
                en_subtitle=self.all_en_subtitles,
                zh_subtitle=self.all_other_subtitles,
                results=self.coarse_alignments,
                season_id=i,
                episode_id=j,
                bias=200,
                zh_split=True
            )
            epi2sub = filter_by_idx(turn_sub2epi_into_epi2sub(alignment_seeds[(i, j)]))
            # Extend the neighbor
            while True:
                temp_len = len(turn_sub2epi_into_epi2sub(epi2sub))
                epi2sub = filter_by_idx(add_neighbor_subtitles_to_episode(en_subset, epi2sub, tbbt_episode))
                if temp_len == len(turn_sub2epi_into_epi2sub(epi2sub)):
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

                if temp_len == len(turn_sub2epi_into_epi2sub(epi2sub)):
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

                if temp_len == len(turn_sub2epi_into_epi2sub(epi2sub)):
                    break

            # Extend Subtitle ids with its min and max index
            for x in epi2sub:
                epi2sub[x] = [i for i in range(min(epi2sub[x]), max(epi2sub[x]) + 1)]

            # Perform ultimate alignment
            gaps = get_final_stage_gap_pairs(epi2sub)
            epi2sub = ultimate_alignment(gaps, epi2sub, en_subset, tbbt_episode)

            further_alignment[(i, j)] = epi2sub

        with open(self.root_path + "1_alignment_extension.pkl", "wb") as f:
            pkl.dump(further_alignment, f)
        return further_alignment

    def add_head_tail(self, alignment_seeds):
        further_alignment = {}
        for (i, j) in alignment_seeds.keys():
            try:
                print(i, j)
                (en_subset, zh_subset, tbbt_episode) = fetch_subsets(
                    episode=self.all_transcripts,
                    en_subtitle=self.all_en_subtitles,
                    zh_subtitle=self.all_other_subtitles,
                    results=self.coarse_alignments,
                    season_id=i,
                    episode_id=j,
                    bias=200,
                    zh_split=True
                )
                epi2sub = filter_by_idx(alignment_seeds[(i, j)])

                head_tail_sub2epi = {}

                gap_pairs = fetch_before_after(en_subset, zh_subset, tbbt_episode, epi2sub)
                for item in gap_pairs:
                    temp = before_after_wer_match(en_subset, tbbt_episode, item[0], item[1])
                    for x in temp:
                        head_tail_sub2epi[x] = temp[x]

                temp = turn_sub2epi_into_epi2sub(head_tail_sub2epi)
                # print(temp)
                if temp != {}:
                    if min(list(turn_sub2epi_into_epi2sub(temp).keys())) < 0:
                        temp = {}
                further_alignment[(i, j)] = filter_by_idx(temp)

                # print(temp)
                # print(filter_by_idx(epi2sub))
                # print("=="*50)
            except:
                print("Pass i j")

        with open(self.root_path + '2_alignment_head_tail.pkl', 'wb') as f:
            pkl.dump(further_alignment, f)

        ultimate_data = {}
        for item in alignment_seeds:
            temp = alignment_seeds[item]
            if item in further_alignment:
                temp = merge_episode_alignment(temp, further_alignment[item])
            ultimate_data[item] = temp

        with open(self.root_path + '2_alignment_ultimate.pkl', 'wb') as f:
            pkl.dump(further_alignment, f)

        return ultimate_data

    def visualization_with_xlxs(self, alignment, xlsx_path):
        for epi_key in tqdm(list(alignment.keys())[:]):
            try:
                generate_xlxs_for_episode(season_id=epi_key[0],
                                          episode_id=epi_key[1],
                                          tbbt_transcripts=self.all_transcripts,
                                          en_subtitle=self.all_en_subtitles,
                                          zh_subtitle=self.all_other_subtitles,
                                          results=self.coarse_alignments,
                                          xlsx_path=xlsx_path,
                                          alignment=alignment,
                                          zh_split=True)
            except:
                print("Pass")
                pass


fine_aligner = FineAligner(
    en_subtitle_path="../../../source_data/subtitles/en_zh/en_subtitles.pkl",
    other_subtitle_path="../../../source_data/subtitles/en_zh/zh_subtitles.pkl",
    transcript_path="../../../source_data/transcripts/tbbt/tbbt_transcripts.pkl",
    coarse_alignment_path="../../coarse_alignment/results/tbbt_en_zh.pkl",
    root_path="../results/tbbt_en_zh/")

seeds = fine_aligner.alignment_seeds_generation()

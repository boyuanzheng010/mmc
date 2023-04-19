import pickle as pkl
import json
from collections import defaultdict
import jiwer
from copy import deepcopy
import re
import string
import xlsxwriter

from .helper_functions import *
from .preprocessing import fetch_subsets

# Set the data cleaning method
transformation = jiwer.Compose([
    jiwer.ToLowerCase(),
    jiwer.RemoveMultipleSpaces(),
    jiwer.ExpandCommonEnglishContractions(),
    jiwer.RemovePunctuation(),
    jiwer.Strip()
])


def generate_xlxs_for_episode(season_id, episode_id, tbbt_transcripts, en_subtitle, zh_subtitle, results, xlsx_path, alignment):
    # Define xlsx file
    episode_book = xlsxwriter.Workbook(xlsx_path + 'episodes/episode_s%de%d.xlsx'%(season_id, episode_id))
    episode_sheet = episode_book.add_worksheet()
    episode_bold = episode_book.add_format({'bold':1})
    for j, item in enumerate(['utterance', 'speaker', 'subtitle id']):
        episode_sheet.write(0, j, item, episode_bold)

    subtitle_book = xlsxwriter.Workbook(xlsx_path + 'subtitles/subtitle_s%de%d.xlsx'%(season_id, episode_id))
    subtitle_sheet = subtitle_book.add_worksheet()
    subtitle_bold = subtitle_book.add_format({'bold':1})
    for j, item in enumerate(['subtitle_en', 'subtitle_zh', 'episode id']):
        subtitle_sheet.write(0, j, item, subtitle_bold)

    # Load Data
    epi2sub = alignment[(season_id, episode_id)]
    sub2epi = turn_sub2epi_into_epi2sub(epi2sub)

    (en_subset, zh_subset, tbbt_episode) = fetch_subsets(
        episode=tbbt_transcripts,
        en_subtitle=en_subtitle,
        zh_subtitle=zh_subtitle,
        results=results,
        season_id=season_id,
        episode_id=episode_id,
        bias=200
    )

    # Write into file
    for i, (utt, speaker) in enumerate(tbbt_episode):
        if i in epi2sub:
            temp = [utt, str(speaker), " ".join([str(item+2) for item in epi2sub[i]])]
            for j, item in enumerate(temp):
                episode_sheet.write(i+1, j, item, episode_bold)
        else:
            temp = [utt, str(speaker), " "]
            for j, item in enumerate(temp):
                episode_sheet.write(i+1, j, item)

    for i, subtitle in enumerate(en_subset):
        if i in sub2epi:
            temp = [subtitle, zh_subset[i], " ".join([str(item+2) for item in sub2epi[i]])]
            for j, item in enumerate(temp):
                subtitle_sheet.write(i+1, j, item, subtitle_bold)
        else:
            temp = [subtitle, zh_subset[i], " "]
            for j, item in enumerate(temp):
                subtitle_sheet.write(i+1, j, item)

    episode_book.close()
    subtitle_book.close()
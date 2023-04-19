import jsonlines
import pickle as pkl


data = []
with open('data/experiment_inputs/dialogue_prob_source_chinese/train.chinese.512.jsonlines', 'r') as f:
    reader = jsonlines.Reader(f)
    for line in reader:
        data.append(line)

















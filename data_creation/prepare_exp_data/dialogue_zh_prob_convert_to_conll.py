import pickle as pkl
from copy import deepcopy
import jsonlines
import json
from utils.my_util import cluster_mentions, remove_speaker_prefix
from utils.conll_processing_util import remove_empty_sentences, cluster_mention_id_index, cluster_mention_id_index_with_prob

do_correction = False
split = "test"

# Load Chinese Alignment Correction Results
with open("data/raw_source/dialogue_zh/all_correction_results.pkl", 'rb') as f:
    correction_result = pkl.load(f)
# Load Mention Id in English side
with open('en_mention_id_cluster.pkl', 'rb') as f:
    en_mention_id_clusters = pkl.load(f)

# Load Source Data for processing
speaker_dict = {}
with open('data/raw_source/dialogue_zh/all_coref_data_en.json', 'r') as f:
    temp = json.load(f)
    for line in temp:
        scene_id = line['scene_id'][:-1]
        speakers = []
        for sent in line['sentences']:
            speakers.append(" ".join(sent[:sent.index(":")]))
        speaker_dict[scene_id] = speakers
split_dict = {"train":[], "dev":[], "test":[]}
with open('data/raw_source/dialogue_zh/dev_finalized.pkl', 'rb') as f:
    temp = pkl.load(f)
    for line in temp:
        split_dict['dev'].append(line['scene_id'])
with open('data/raw_source/dialogue_zh/test_finalized.pkl', 'rb') as f:
    temp = pkl.load(f)
    for line in temp:
        split_dict['test'].append(line['scene_id'])
with open('data/raw_source/dialogue_zh/train_finalized.pkl', 'rb') as f:
    temp = pkl.load(f)
    for line in temp:
        split_dict['train'].append(line['scene_id'])


data = []
all_ids = []
# with open('data/raw_source/dialogue_zh/all_coref_data_en_zh_finalized.json', 'r') as f:
with open('data/raw_source/dialogue_proj_zh/all_coref_data_en_zh_finalized_word_prob.json', 'r') as f:
    reader = jsonlines.Reader(f)
    for bulk in reader:
        for idx, instance in enumerate(bulk):
            scene_id = instance['scene_id']
            if scene_id == "":
                continue
            sentences = instance['sentences']
            annotations = instance['annotations']
            all_ids.append(scene_id)
            speakers = speaker_dict[scene_id]
            answers = []
            for item in annotations:
                # Correct query
                query = (item['query']['sentenceIndex'], item['query']['startToken'], item['query']['endToken'], item['query']['mention_id'], item['query']['align_prob'])
                antecedents = item['antecedents']
                if antecedents in [['n', 'o', 't', 'P', 'r', 'e', 's', 'e', 'n', 't'], ['null_projection'], ['empty_subtitle']]:
                    answers.append([query, "notPresent"])
                else:
                    temp_answer = []
                    for antecedent in antecedents:
                        if isinstance(antecedent, dict):
                            temp_answer.append((antecedent['sentenceIndex'], antecedent['startToken'], antecedent['endToken'], antecedent['mention_id'], item['query']['align_prob']))
                        else:
                            temp_answer = " ".join(antecedents)
                    answers.append([query, temp_answer])
            #
            # # Add correction results
            # if (scene_id in correction_result) and do_correction:
            #     to_correct = correction_result[scene_id]
            #     correction_dict = to_correct['correction_dict']
            #     remove_set = to_correct['remove_set']
            #     add_dict = to_correct['add_dict']
            #     # Deal with to add_dict
            #     for item in add_dict:
            #         temp = [
            #             (add_dict[item][0], add_dict[item][1], add_dict[item][2], item, add_dict[item][4]),
            #             'notPresent'
            #         ]
            #         answers.append(temp)

            data.append(remove_empty_sentences({
                "sentences": sentences,
                "answers": answers,
                "speakers": speakers,
                "scene_id": scene_id
            }))

all_projection_probability = {}
document = []
for i in range(len(data)):
    sample = data[i]
    if sample['scene_id'] not in split_dict[split]:
        continue
    sentences = sample['sentences']
    speakers = sample['speakers']
    scene_id = sample['scene_id']
    clusters, cluster_probability = cluster_mention_id_index_with_prob(sample['answers'], sentences, en_mention_id_clusters[scene_id], correction_result, do_correction)
    all_projection_probability[sample['scene_id']] = cluster_probability
    part = int(scene_id[7:9])
    begin_line = "#begin document " + "(" + scene_id + "); part " + "%03d" % part
    end_line = "#end document"

    # Prepare for clustering
    cluster_field = []
    for sent in sentences:
        cluster_field.append([""]*len(sent))
    # Add start
    for idx, cluster in enumerate(clusters):
        for sent_id, start, end in cluster:
            end = end - 1
            if start != end:
                # print(cluster_field[sent_id])
                # print(sent_id, start, end, len(sentences[sent_id]))
                # print(sentences[sent_id])
                if cluster_field[sent_id][start] == "":
                    cluster_field[sent_id][start] += "(" + str(idx)
                else:
                    cluster_field[sent_id][start] += "|" + "(" + str(idx)
    # Add start==end
    for idx, cluster in enumerate(clusters):
        for sent_id, start, end in cluster:
            end = end - 1
            if start == end:
                if cluster_field[sent_id][start] == "":
                    cluster_field[sent_id][start] += "(" + str(idx) + ")"
                else:
                    cluster_field[sent_id][start] += "|" + "(" + str(idx) + ")"
    # Add End
    for idx, cluster in enumerate(clusters):
        for sent_id, start, end in cluster:
            end = end - 1
            if start != end:
                try:
                    if cluster_field[sent_id][end] == "":
                        cluster_field[sent_id][end] += str(idx) + ")"
                    else:
                        cluster_field[sent_id][end] += "|" + str(idx) + ")"
                except:
                    print("ERROR")
                    pass
                # if cluster_field[sent_id][end] == "":
                #     cluster_field[sent_id][end] += str(idx) + ")"
                # else:
                #     cluster_field[sent_id][end] += "|" + str(idx) + ")"

    # Build document
    document.append(begin_line + "\n")
    for sent, speaker, cluster_value in zip(sentences, speakers, cluster_field):
        for j, word in enumerate(sent):
            cluster_id = cluster_value[j]
            if cluster_id == "":
                cluster_id = "-"
            temp = [scene_id, str(part), str(j), word, "na", "na", "na", "na", "na", speaker, "na", "na", "na", cluster_id]
            document.append(" ".join(temp)+ "\n")
        document.append("" + "\n")
    document.append(end_line + "\n")

with open("data/conll_style/dialogue_prob_source_chinese/"+ split+'.chinese.v4_gold_conll', 'w') as f:
    f.writelines(document)

with open("data/conll_style/dialogue_prob_source_chinese/"+ split+'_projection_probability.pkl', 'wb') as f:
    pkl.dump(all_projection_probability, f)






























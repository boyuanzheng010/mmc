import pickle as pkl
from copy import deepcopy
import jsonlines
from utils.my_util import cluster_mentions, remove_speaker_prefix
import json


split = "test"


def remove_empty_sentences(instance):
    sentences = instance['sentences']
    answers = instance['answers']
    speakers = instance['speakers']

    # Build old sent_id to new sent_id map
    map_sent_id = {}
    count = 0
    for i, sent in enumerate(sentences):
        if sent == []:
            continue
        map_sent_id[i] = count
        count += 1

    # Collect answers, speakers for each sentence
    temp = []
    for i, sent in enumerate(sentences):
        if sent == []:
            continue
        annotations = []
        for answer in answers:
            if answer[0][0]==i:
                annotations.append(answer)
        temp.append([sent, annotations, speakers[i]])

    # Change Sentence ID
    sentences = []
    answers = []
    speakers = []
    for i, (sent, annotations, speaker) in enumerate(temp):
        # print(i, speaker, sent)
        sentences.append(sent)
        temp_answers = []
        for query, antecedents in annotations:
            new_query = tuple((map_sent_id[query[0]], query[1], query[2], query[3]))
            # print(query, new_query)
            new_antecedents = []
            if isinstance(antecedents, str):
                new_antecedents = antecedents
                # print(new_antecedents)
            else:
                # print(antecedents)
                for antecedent in antecedents:
                    new_antecedents.append((map_sent_id[antecedent[0]], antecedent[1], antecedent[2], antecedent[3]))
            # print(new_antecedents)
            temp_answers.append([new_query, new_antecedents])
        answers.extend(temp_answers)
        speakers.append(speaker)

    return {
        "sentences": sentences,
        "answers": answers,
        "speakers": speakers,
        "scene_id": instance['scene_id']
    }


def cluster_mention_id_index(answers, sentences, en_clusters, correction_result):
    """
    We cluster Farsi Side mentions according to the index in English Side
    """
    # Collect Mention_ID to Chinese Side tuple
    zh_mention_dict = {}
    for answer in answers:
        query = answer[0]
        zh_mention_dict[query[3]] = (query[0], query[1], query[2])
        antecedents = answer[1]
        if isinstance(antecedents, list):
            for antecedent in antecedents:
                zh_mention_dict[antecedent[3]] = (antecedent[0], antecedent[1], antecedent[2])


    # Incorporate Maunal Correction
    scene_id = list(zh_mention_dict.keys())[0]
    if scene_id[:10] in correction_result:
        to_correct = correction_result[scene_id[:10]]
        correction_dict = to_correct['correction_dict']
        remove_set = to_correct['remove_set']
        source_zh_mention_dict = deepcopy(zh_mention_dict)
        zh_mention_dict = {}

        # Perform Correction
        for mention_id in source_zh_mention_dict:
            # remove mentions
            if mention_id in remove_set:
                continue
            # Correct start, end
            elif mention_id in correction_dict:
                zh_mention_dict[mention_id] = tuple([source_zh_mention_dict[mention_id][0], correction_dict[mention_id][1], correction_dict[mention_id][2]])
            else:
                zh_mention_dict[mention_id] = source_zh_mention_dict[mention_id]

    # Gather Chinese Side cluster according to en_clusters
    new_cluster = []
    for cluster in en_clusters:
        temp = []
        for mention_id in cluster:
            if mention_id in zh_mention_dict:
                temp.append(zh_mention_dict[mention_id])
        if temp:
            new_cluster.append(temp)

    # Merge Cluster using (sent_id, start_id, end_id)
    all_clusters = deepcopy(new_cluster)
    merged_clusters = []
    for cluster in all_clusters:
        existing = None
        for mention in cluster:
            for merged_cluster in merged_clusters:
                if mention in merged_cluster:
                    existing = merged_cluster
                    break
            if existing is not None:
                break
        if existing is not None:
            existing.update(cluster)
        else:
            merged_clusters.append(set(cluster))
    merged_clusters = [list(cluster) for cluster in merged_clusters]
    # print(len(merged_clusters))
    # for item in merged_clusters:
    #     print(item)
    # print("=="*50)

    return merged_clusters


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

# with open('split_dict.pkl', 'wb') as f:
#     pkl.dump(split_dict, f)

with open('en_mention_id_cluster.pkl', 'rb') as f:
    en_mention_id_clusters = pkl.load(f)

with open("data/raw_source/dialogue_fa/correction_results.pkl", 'rb') as f:
    correction_result = pkl.load(f)
"""
Cluster Chinese Mentions using English-Side Mention_IDs
"""
data = []
all_ids = []
with open('data/raw_source/dialogue_fa/all_coref_data_en_fa_finalized.json', 'r') as f:
# with open('data/raw_source/dialogue_zh/dev-test-batch1_zh.json', 'r') as f:
    reader = jsonlines.Reader(f)
    for bulk in reader:
        for idx, instance in enumerate(bulk):
            # if idx>=10:
            #     break
            scene_id = instance['scene_id']
            print(scene_id)
            if scene_id == "":
                continue
            sentences = instance['sentences']
            # print(sentences)
            # sentences = [[token for token in "".join(sent)] for sent in instance['sentences']]
            annotations = instance['annotations']
            all_ids.append(scene_id)
            speakers = speaker_dict[scene_id]
            answers = []
            for item in annotations:
                query = (item['query']['sentenceIndex'], item['query']['startToken'], item['query']['endToken'], item['query']['mention_id'])
                antecedents = item['antecedents']
                if antecedents in [['n', 'o', 't', 'P', 'r', 'e', 's', 'e', 'n', 't'], ['null_projection'], ['empty_subtitle']]:
                    answers.append([query, "notPresent"])
                else:
                    temp_answer = []
                    for antecedent in antecedents:
                        if isinstance(antecedent, dict):
                            temp_answer.append((antecedent['sentenceIndex'], antecedent['startToken'], antecedent['endToken'], antecedent['mention_id']))
                        else:
                            temp_answer = " ".join(antecedents)
                    answers.append([query, temp_answer])

            # Add correction results
            if scene_id in correction_result:
                to_correct = correction_result[scene_id]
                correction_dict = to_correct['correction_dict']
                remove_set = to_correct['remove_set']
                add_dict = to_correct['add_dict']
                # Deal with to add_dict
                for item in add_dict:
                    temp = [
                        (add_dict[item][0], add_dict[item][1], add_dict[item][2], item),
                        'notPresent'
                    ]
                    answers.append(temp)

            data.append(remove_empty_sentences({
                "sentences": sentences,
                "answers": answers,
                "speakers": speakers,
                "scene_id": scene_id
            }))

document = []
for i in range(len(data)):
    # if i > 2:
    #     continue
    sample = data[i]
    if sample['scene_id'] not in split_dict[split]:
        continue
    # print(sample)
    # print()
    # original_sentences = sample['sentences']
    # original_clusters = cluster_mentions(sample['answers'], original_sentences)
    # sentences, clusters, speakers = remove_speaker_prefix(original_sentences, original_clusters)
    sentences = sample['sentences']
    # clusters = cluster_mentions(sample['answers'], sentences)
    speakers = sample['speakers']
    scene_id = sample['scene_id']
    clusters = cluster_mention_id_index(sample['answers'], sentences, en_mention_id_clusters[scene_id], correction_result)
    # print(scene_id)
    # for item in clusters:
    #     print(item)
    # print("=="*50)
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
                    print("Wrong")
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

with open("data/conll_style/dialogue_corrected_farsi/"+ split+'.farsi.v4_gold_conll', 'w') as f:
    f.writelines(document)

print(len(document))








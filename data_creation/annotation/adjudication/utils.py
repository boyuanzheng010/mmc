import csv
import json
from copy import deepcopy
import pickle as pkl
csv.field_size_limit(1131072)


from annotation.analysis.utils.data_util import generate_all_clusters_combine_speakers
from annotation.analysis.utils.data_util import generate_clusters_no_plural_combine_speakers
from annotation.analysis.utils.data_util import collect_mentions


def read_turkle_annotation_multiple_scene(path):
    """
    Load the annotation_result along with the document
    Output: sentence along with all annotations

    In this version, the input contains annotations from multiple scenes
    Different scenes could be split with sentence_offsets and query_offsets
    """
    output = []
    with open(path, 'r') as f:
        annotation_reader = csv.DictReader(f)
        for instance in annotation_reader:
            inputs = json.loads(instance["Input.json_data"])
            all_sentences = inputs['sentences']
            all_query_spans = inputs['querySpans']
            sentence_offsets = inputs['sentence_offsets']
            query_spans_offsets = inputs['querySpans_offsets']
            all_scene_ids = inputs['scene_ids']
            all_answer_spans = json.loads(instance["Answer.answer_spans"])

            for i in range(len(sentence_offsets) - 1):
                sent_start, sent_end = sentence_offsets[i], sentence_offsets[i + 1]
                query_start, query_end = query_spans_offsets[i], query_spans_offsets[i + 1]
                original_sentences = all_sentences[sent_start: sent_end]
                sentences = []
                for item in original_sentences:
                    temp_sent = []
                    if isinstance(item[0], list):
                        temp_sent.extend(item[0])
                        temp_sent.extend(item[1:])
                        sentences.append(temp_sent)
                    else:
                        sentences.append(item)
                query_spans = all_query_spans[query_start: query_end]
                for item in query_spans:
                    item['sentenceIndex'] -= sentence_offsets[i]

                answer_spans = all_answer_spans[query_start: query_end]
                for item in answer_spans:
                    item['querySpan']['sentenceIndex'] -= sentence_offsets[i]
                    for x in item['span_list']:
                        if x['startToken'] == -1 and x['endToken'] == -1:
                            continue
                        x['sentenceIndex'] -= sentence_offsets[i]

                temp = {
                    "sentences": sentences,
                    "query_spans": query_spans,
                    "answer_spans": answer_spans,
                    "WorkerId": instance['WorkerId'],
                    "scene_id": all_scene_ids[i],
                }
                # Collect mentions into clusters
                temp['clusters_all'] = generate_all_clusters_combine_speakers(temp)
                temp['clusters_no_plural'] = generate_clusters_no_plural_combine_speakers(temp)

                # Collect mentions (For Kappa Cohen)
                answers = collect_mentions(temp)
                temp["answers"] = answers

                # Add to output
                output.append(temp)

    return output

def gather_by_scene(annotations):
    """
    annotations: [sentence along with all annotation_result]
    return {scene_key: [annotations]}
    """
    output = {}
    for instance in annotations:
        key_id = instance['scene_id']
        if key_id not in output:
            output[key_id] = [instance]
        else:
            output[key_id].append(instance)
    return output



def extract_common_cluster(anno1, anno2):
    clusters = []
    for i in range(len(anno1)):
        if anno1[i]!=anno2[i]:
            continue
        query = "_".join([str(x) for x in anno1[i][0]])
        answer = "|".join(anno1[i][1])
        if answer == "notMention":
            continue

        to_adds = [query]
        if answer != "notPresent":
            to_adds.append(answer)
        signal = True
        for cluster in clusters:
            if len(set(to_adds)&set(cluster))!=0:
                cluster.extend(to_adds)
                signal = False
        if signal:
            clusters.append(to_adds)

    for i in range(len(clusters)):
        clusters[i] = sorted(list(set(clusters[i])))

    return clusters

def add_to_common_cluster(clusters, anno1, anno2):
    for i in range(len(anno1)):
        if anno1[i]==anno2[i]:
            continue
        query = "_".join([str(x) for x in anno1[i][0]])
        answer1 = "|".join(anno1[i][1])
        answer2 = "|".join(anno2[i][1])
        if (len(answer1.strip().split('_'))!=3) or (len(answer2.strip().split('_'))!=3):
            continue

        label1 = -1
        label2 = -1
        for k, cluster in enumerate(clusters):
            if answer1 in cluster:
                label1 = k
            if answer2 in cluster:
                label2 = k
        if label1 == label2:
            clusters[label1].extend([query, answer1, answer2])

    for i in range(len(clusters)):
        clusters[i] = sorted(list(set(clusters[i])))

    return clusters


def flatten_cluster(clusters):
    output = []
    for cluster in clusters:
        output.extend(cluster)
    return output

def analyze_difference(clusters, anno1, anno2):
    count = 0
    for i in range(len(anno1)):
        query = "_".join([str(x) for x in anno1[i][0]])
        answer1 = "|".join(anno1[i][1])
        answer2 = "|".join(anno2[i][1])

        # Initialize labels for two answers
        label1 = -1
        label2 = -1
        if answer1 == "notPresent":
            label1 = -2
        if answer1 == "notMention":
            label1 = -3
        if answer2 == "notPresent":
            label2 = -2
        if answer2 == "notMention":
            label2 = -3

        for k, cluster in enumerate(clusters):
            if answer1 in cluster:
                label1 = k
            if answer2 in cluster:
                label2 = k

        if label1==label2:
            count += 1
    return [len(anno1), count, int(count/len(anno1)*100)]


def analyze_types_of_difference(clusters, anno1, anno2):
    count = 0
    for i in range(len(anno1)):
        query = "_".join([str(x) for x in anno1[i][0]])
        answer1 = "|".join(anno1[i][1])
        answer2 = "|".join(anno2[i][1])

        # Initialize labels for two answers
        label1 = -1
        label2 = -1
        if answer1 == "notPresent":
            label1 = -2
        if answer1 == "notMention":
            label1 = -3
        if answer2 == "notPresent":
            label2 = -2
        if answer2 == "notMention":
            label2 = -3

        for k, cluster in enumerate(clusters):
            if answer1 in cluster:
                label1 = k
            if answer2 in cluster:
                label2 = k

        if label1!=label2:
            if label1  in [-3]:
                if label2  in [-2]:
                    count += 1
    return [len(anno1), count, int(count/len(anno1)*100)]



def get_disagreement_types(anno1, anno2):
    """
    Return the annotation type in for each annotation
    -1: Not in Cluster
    -2: No Previous Mention
    -3: Not Mention
    0-len(clusters): cluster id
    """
    # Build Clusters
    common_clusters = extract_common_cluster(anno1, anno2)
    clusters = add_to_common_cluster(common_clusters, anno1, anno2)

    all_label1 = []
    all_label2 = []
    for i in range(len(anno1)):
        query = "_".join([str(x) for x in anno1[i][0]])
        answer1 = "|".join(anno1[i][1])
        answer2 = "|".join(anno2[i][1])

        # Initialize labels for two answers
        label1 = -1
        label2 = -1
        if answer1 == "notPresent":
            label1 = -2
        if answer1 == "notMention":
            label1 = -3
        if answer2 == "notPresent":
            label2 = -2
        if answer2 == "notMention":
            label2 = -3

        for k, cluster in enumerate(clusters):
            if answer1 in cluster:
                label1 = k
            if answer2 in cluster:
                label2 = k

        all_label1.append(label1)
        all_label2.append(label2)
    return all_label1, all_label2


def analyze_types_of_difference(clusters, anno1, anno2):
    count = 0
    for i in range(len(anno1)):
        query = "_".join([str(x) for x in anno1[i][0]])
        answer1 = "|".join(anno1[i][1])
        answer2 = "|".join(anno2[i][1])

        # Initialize labels for two answers
        label1 = -1
        label2 = -1
        if answer1 == "notPresent":
            label1 = -2
        if answer1 == "notMention":
            label1 = -3
        if answer2 == "notPresent":
            label2 = -2
        if answer2 == "notMention":
            label2 = -3

        for k, cluster in enumerate(clusters):
            if answer1 in cluster:
                label1 = k
            if answer2 in cluster:
                label2 = k

        if label1!=label2:
            if label1  in [-3]:
                if label2  in [-2]:
                    count += 1
    return [len(anno1), count, int(count/len(anno1)*100)]



def get_error_matrix(label1, label2):
    # Define slice range for three types of spans
    temp_range = list(set(label1+label2))
    temp_range.append(-1)
    range_span = [token for token in temp_range if token not in [-2, -3]]
    range_not_mention = [-3]
    range_not_previous = [-2]

    # Define Number List
    span_span = []
    span_not_previous = []
    span_not_mention = []
    not_previous_span = []
    not_previous_not_mention = []
    not_mention_span = []
    not_mention_not_previous = []

    # Count Numbers
    for i, (a, b) in enumerate(zip(label1, label2)):
        # Only count not agreed annotations
        if a == b:
            continue

        # Add to list
        if (a in range_span) and (b in range_span):
            span_span.append(i)

        if (a in range_span) and (b in range_not_previous):
            span_not_previous.append(i)

        if (a in range_span) and (b in range_not_mention):
            span_not_mention.append(i)

        if (a in range_not_previous) and (b in range_span):
            not_previous_span.append(i)

        if (a in range_not_previous) and (b in range_not_mention):
            not_previous_not_mention.append(i)

        if (a in range_not_mention) and (b in range_span):
            not_mention_span.append(i)

        if (a in range_not_mention) and (b in range_not_previous):
            not_mention_not_previous.append(i)

    return [
        span_span,
        span_not_previous,
        span_not_mention,
        not_previous_span,
        not_previous_not_mention,
        not_mention_span,
        not_mention_not_previous
    ]







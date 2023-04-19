import csv
import json

csv.field_size_limit(1131072)


def collect_mentions(instance):
    """
    Collect answer mentions
    """
    answer_spans = instance['answer_spans']
    mentions = []
    for item in answer_spans:
        # Process annotation into tuple
        query = (item['querySpan']['sentenceIndex'], item['querySpan']['startToken'], item['querySpan']['endToken'])
        answers = []
        for answer in item['span_list']:
            answers.append((answer['sentenceIndex'], answer['startToken'], answer['endToken']))

        if item['notMention']:
            mentions.append([query, "notMention"])
        elif item['notPresent']:
            mentions.append([query, "notPresent"])
        else:
            mentions.append([query, answers])
    return mentions


def generate_all_clusters(instance):
    """
    Collect mentions into clusters
    We treat plural as a independent cluster
    """
    answer_spans = instance['answer_spans']
    clusters = []
    for item in answer_spans:
        # Process annotation into tuple
        query = []
        query.append(
            "_".join([str(item['querySpan']['sentenceIndex']), str(item['querySpan']['startToken']),
                      str(item['querySpan']['endToken'])])
        )
        query = tuple(query)

        answers = []
        for answer in item['span_list']:
            answers.append(
                "_".join([str(answer['sentenceIndex']), str(answer['startToken']), str(answer['endToken'])])
            )
        answers = tuple(answers)

        if item['notMention']:
            continue
        else:
            to_adds = [query]
            if not item['notPresent']:
                to_adds.append(answers)
            # Add to clusters
            signal = True
            for cluster in clusters:
                if answers in cluster:
                    cluster.extend(to_adds)
                    signal = False
            if signal:
                clusters.append(to_adds)

    output = []
    for item in clusters:
        output.append(set(item))
        # output.append(item)
    return output

def generate_all_clusters_combine_speakers(instance):
    """
    Collect mentions into clusters
    We treat plural as an independent cluster

    In this version, we cluster each speaker into a cluster.
    To achieve this by the following steps:
    1.Initialize one cluster for each speaker and use all_speaker_mentions to record all speaker
    2.Build cluster using all mentions. Remove mention from all_speaker_mentions if it is hit
    3.Iterate all clusters and pop out remaining mention node in all_speaker_mentions
    """
    answer_spans = instance['answer_spans']
    sentences = instance['sentences']
    clusters = []

    # Initialize clusters with all speaker mention clusters
    speaker_clusters = {}
    for i, sent in enumerate(sentences):
        startToken = 0
        endToken = sent.index(":")
        speaker = " ".join(sent[startToken: endToken])
        if speaker not in speaker_clusters:
            speaker_clusters[speaker] = ["_".join([str(i), str(startToken), str(endToken)])]
        else:
            speaker_clusters[speaker].append("_".join([str(i), str(startToken), str(endToken)]))
    # print(speaker_clusters)


    for item in answer_spans:
        # Process annotation into tuple
        query = []
        query.append(
            "_".join([str(item['querySpan']['sentenceIndex']), str(item['querySpan']['startToken']),
                      str(item['querySpan']['endToken'])])
        )
        query = tuple(query)

        answers = []
        for answer in item['span_list']:
            token = "_".join([str(answer['sentenceIndex']), str(answer['startToken']), str(answer['endToken'])])
            for speaker in speaker_clusters:
                if token in speaker_clusters[speaker]:
                    token = speaker
            answers.append(token)
        answers = tuple(answers)

        if item['notMention']:
            continue
        else:
            to_adds = [query]
            if not item['notPresent']:
                to_adds.append(answers)
            # Add to clusters
            signal = True
            for cluster in clusters:
                if answers in cluster:
                    cluster.extend(to_adds)
                    signal = False
            if signal:
                clusters.append(to_adds)

    output = []
    for cluster in clusters:
        if cluster:
            output.append(set(cluster))
    return output


def generate_clusters_no_plural_combine_speakers(instance):
    """
    Collect mentions into clusters
    We treat plural as an independent cluster

    In this version, we cluster each speaker into a cluster.
    To achieve this by the following steps:
    1.Initialize one cluster for each speaker and use all_speaker_mentions to record all speaker
    2.Build cluster using all mentions. Remove mention from all_speaker_mentions if it is hit
    3.Iterate all clusters and pop out remaining mention node in all_speaker_mentions
    """
    answer_spans = instance['answer_spans']
    sentences = instance['sentences']
    clusters = []

    # Initialize clusters with all speaker mention clusters
    speaker_clusters = {}
    for i, sent in enumerate(sentences):
        startToken = 0
        endToken = sent.index(":")
        speaker = " ".join(sent[startToken: endToken])
        if speaker not in speaker_clusters:
            speaker_clusters[speaker] = ["_".join([str(i), str(startToken), str(endToken)])]
        else:
            speaker_clusters[speaker].append("_".join([str(i), str(startToken), str(endToken)]))
    # print(speaker_clusters)


    for item in answer_spans:
        # Process annotation into tuple
        query = []
        query.append(
            "_".join([str(item['querySpan']['sentenceIndex']), str(item['querySpan']['startToken']),
                      str(item['querySpan']['endToken'])])
        )
        query = tuple(query)

        answers = []
        for answer in item['span_list']:
            token = "_".join([str(answer['sentenceIndex']), str(answer['startToken']), str(answer['endToken'])])
            for speaker in speaker_clusters:
                if token in speaker_clusters[speaker]:
                    token = speaker
            answers.append(token)
        answers = tuple(answers)

        if item['notMention']:
            continue
        else:
            to_adds = [query]
            if (not item['notPresent']) and (len(answers) == 1):
                to_adds.append(answers)
            # Add to clusters
            signal = True
            for cluster in clusters:
                if answers in cluster:
                    cluster.extend(to_adds)
                    signal = False
            if signal:
                clusters.append(to_adds)

    output = []
    for cluster in clusters:
        if cluster:
            output.append(set(cluster))
    return output


def read_annotation(path):
    """
    Load the annotation along with the document
    Output: sentence along with all annotations
    """
    output = []
    with open(path, 'r') as f:
        annotation_reader = csv.DictReader(f)
        for instance in annotation_reader:
            temp = {}
            for item in instance:
                if item == "Input.json_data":
                    temp['sentences'] = json.loads(instance[item])['sentences']
                elif item == "Answer.answer_spans":
                    temp["answer_spans"] = json.loads(instance[item])
                else:
                    temp[item] = instance[item]
            # Collect mentions into clusters
            temp['clusters_all'] = generate_all_clusters_combine_speakers(temp)
            temp['clusters_no_plural'] = generate_clusters_no_plural_combine_speakers(temp)

            # Collect mentions (For Kappa Cohen)
            answers = collect_mentions(temp)
            temp["answers"] = answers

            # Add to output
            output.append(temp)

    return output


def read_turkle_annotation_multiple_scene(path):
    """
    Load the annotation along with the document
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
            all_answer_spans = json.loads(instance["Answer.answer_spans"])

            for i in range(len(sentence_offsets)-1):
                sent_start, sent_end = sentence_offsets[i], sentence_offsets[i+1]
                query_start, query_end = query_spans_offsets[i], query_spans_offsets[i+1]
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
                        if x['startToken']==-1 and x['endToken']==-1:
                            continue
                        x['sentenceIndex'] -= sentence_offsets[i]

                temp = {
                    "sentences": sentences,
                    "query_spans": query_spans,
                    "answer_spans": answer_spans,
                    "WorkerId": instance['WorkerId']
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


def read_mturk_annotation_multiple_scene(path):
    """
    Load the annotation along with the document
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
                    "WorkerId": instance['WorkerId']
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

def gather_by_annotator(annotations):
    """
    annotations: [sentence along with all annotation]
    return {username: [annotations]}
    """
    output = {}
    for instance in annotations:
        key_id = instance['Turkle.Username']
        if key_id not in output:
            output[key_id] = [instance]
        else:
            output[key_id].append(instance)
    return output


def gather_by_scene(annotations):
    """
    annotations: [sentence along with all annotation]
    return {scene_key: [annotations]}
    """
    output = {}
    for instance in annotations:
        key_id = "|".join(instance['sentences'][0][1:10])
        if key_id not in output:
            output[key_id] = [instance]
        else:
            output[key_id].append(instance)
    return output

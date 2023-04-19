from copy import deepcopy


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
            if answer[0][0] == i:
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


def cluster_mention_id_index(answers, sentences, en_clusters, correction_result, do_correction=False):
    """
    We cluster Chinese Side mentions according to the index in English Side
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
    if (scene_id[:10] in correction_result) and do_correction:
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
                zh_mention_dict[mention_id] = tuple(
                    [source_zh_mention_dict[mention_id][0], correction_dict[mention_id][1],
                     correction_dict[mention_id][2]])
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
    return merged_clusters


def cluster_mention_id_index_with_prob(answers, sentences, en_clusters, correction_result, do_correction=False):
    """
    We cluster Chinese Side mentions according to the index in English Side
    """
    # Collect Mention_ID to Chinese Side tuple
    zh_mention_dict = {}
    id_to_prob_map = {}
    for answer in answers:
        query = answer[0]
        zh_mention_dict[query[3]] = (query[0], query[1], query[2])
        id_to_prob_map[query[3]] = query[4]
        antecedents = answer[1]
        if isinstance(antecedents, list):
            for antecedent in antecedents:
                zh_mention_dict[antecedent[3]] = (antecedent[0], antecedent[1], antecedent[2])
                id_to_prob_map[antecedent[3]] = antecedent[4]


    # Incorporate Maunal Correction
    if do_correction:
        scene_id = list(zh_mention_dict.keys())[0]
        if (scene_id[:10] in correction_result) and do_correction:
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
                    zh_mention_dict[mention_id] = tuple(
                        [source_zh_mention_dict[mention_id][0], correction_dict[mention_id][1],
                        correction_dict[mention_id][2]])
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

    # Generate Probability aligned with clusters
    idxs_to_mention = {}
    for mention in zh_mention_dict:
        idxs_to_mention[zh_mention_dict[mention]] = mention
    all_probability = []
    for cluster in merged_clusters:
        temp = []
        for mention in cluster:
            temp.append(id_to_prob_map[idxs_to_mention[mention]])
        all_probability.append(temp)
    return merged_clusters, all_probability

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
            if answer[0][0] == i:
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
            new_query = tuple((map_sent_id[query[0]], query[1], query[2], query[3], query[4]))
            # print(query, new_query)
            new_antecedents = []
            if isinstance(antecedents, str):
                new_antecedents = antecedents
                # print(new_antecedents)
            else:
                # print(antecedents)
                for antecedent in antecedents:
                    new_antecedents.append((map_sent_id[antecedent[0]], antecedent[1], antecedent[2], antecedent[3], antecedent[4]))
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

import pickle as pkl
from copy import deepcopy


def cluster_mentions(answers, sentences):
    """
    Cluster mention including plural. The clustering steps are as follows:
    1. Gather all non-plural query-annotation pairs
    2. Merge all non-plural pairs to build big cluster
    3. Add Plural: turn each mention in plural into {1.speaker name, 2.cluster id, 3.turn sent_id, start, end into special identification}, then use these to build a string for clustering
    4. Add each pair to cluster and do merging
    5. Remove strings from each cluster
    ?One Thing to Consider: Whether use speaker mention in sentence to merge to speaker cluster?
    """
    all_clusters = []
    speaker_set = set()
    # Generate all cluster (no plural), we will add plural latter
    for query, annotations in answers:
        if isinstance(annotations, str):
            if annotations=="notPresent":
                all_clusters.append([query])
        elif len(annotations)==1:
            temp = [query]
            for token in annotations:
                if token[1]==0:
                    try:
                        speaker = " ".join(sentences[token[0]][token[1]: sentences[token[0]].index(":")]).lower()
                        temp.append(speaker)
                        speaker_set.add(speaker)
                    except:
                        continue
                else:
                    temp.append(token)
            all_clusters.append(temp)

    # Merge clusters if any clusters have common mentions
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

    # Add Plural
    for query, annotations in answers:
        if isinstance(annotations, list):
            if len(annotations)>1:
                temp_anno = []
                for token in annotations:
                    if token[1]==0:
                        try:
                            speaker = " ".join(sentences[token[0]][token[1]: sentences[token[0]].index(":")]).lower()
                            temp_anno.append(speaker)
                            speaker_set.add(speaker)
                        except:
                            continue
                    else:
                        # If the cluster is already in cluster, use the cluster id as identification, else use the index
                        cluster_idx = -1
                        for idx, cluster in enumerate(merged_clusters):
                            if token in cluster:
                                cluster_idx = idx
                                break
                        if cluster_idx != -1:
                            temp_anno.append(str(cluster_idx))
                        else:
                            temp_anno.append("*" + "*".join([str(num) for num in token]) + "*")
                temp_cluster = [query, "||".join(sorted(temp_anno))]
                merged_clusters.append(temp_cluster)

    # Merge Plural
    all_clusters = deepcopy(merged_clusters)
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

    output = []
    for cluster in merged_clusters:
        output.append([token for token in cluster if isinstance(token, tuple)])

    return output


def remove_speaker_prefix(sentences, clusters):
    """
    We remove the speaker prefix in both sentence and cluster offsets
    """
    new_cluster = []
    for cluster in clusters:
        temp = []
        for token in cluster:
            offset = sentences[token[0]].index(":")+1
            temp.append((token[0], token[1]-offset, token[2]-offset))
        new_cluster.append(temp)

    new_sentences = []
    speakers = []
    for sent in sentences:
        new_sentences.append(sent[sent.index(":")+1:])
        speakers.append(" ".join(sent[:sent.index(":")]))

    return new_sentences, new_cluster, speakers










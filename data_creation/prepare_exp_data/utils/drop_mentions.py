import jsonlines
import json
import random


def flatten_clusters(clusters):
    all_mentions = []
    for cluster in clusters:
        all_mentions.extend(cluster)
    return all_mentions


def generate_dropped_clusters_ontonotes(clusters, drop_ratio=0.2):
    all_mentions = flatten_clusters(clusters)
    drop_mentions = random.sample(population=all_mentions, k=int(len(all_mentions) * drop_ratio))
    new_clusters = []
    for cluster in clusters:
        temp_cluster = []
        for mention in cluster:
            if mention not in drop_mentions:
                temp_cluster.append(mention)
        if len(temp_cluster) > 1:
            new_clusters.append(temp_cluster)
    return new_clusters


def generate_dropped_clusters_mmc(clusters, drop_ratio=0.2):
    all_mentions = flatten_clusters(clusters)
    drop_mentions = random.sample(population=all_mentions, k=int(len(all_mentions) * drop_ratio))
    new_clusters = []
    for cluster in clusters:
        temp_cluster = []
        for mention in cluster:
            if mention not in drop_mentions:
                temp_cluster.append(mention)
        if len(temp_cluster) > 0:
            new_clusters.append(temp_cluster)
    return new_clusters

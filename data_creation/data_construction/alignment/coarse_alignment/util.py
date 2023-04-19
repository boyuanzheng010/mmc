
def get_segments(episode_key, all_data):
    """
    Collect all utterances in a episode defined by episode_key
    The output is a list of utterances in the defined episode
    """
    segments = []
    for x in all_data:
        if x == episode_key:
            # Collect Segments
            print(all_data[x])
            for utt in all_data[x]['sentences_transformed']:
                tokens = utt.strip().split(' ')
                length = len(tokens)
                num_iter = length // 6
                for i in range(num_iter):
                    segments.append(" ".join(tokens[i * 6: i * 6 + 6]))
    return segments
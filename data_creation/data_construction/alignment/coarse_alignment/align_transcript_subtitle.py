from tqdm import tqdm
import pickle as pkl
from util import get_segments

# Set file path
transcript_path = "../../source_data/transcripts/friends/friends_transcripts.pkl"
en_subtitle_path = "../../source_data/subtitles/en_zh/en_subtitles_transformed.pkl"
other_subtitle_path = "../../source_data/subtitles/en_zh/zh_subtitles.pkl"
output_path = "results/friends_en_zh_coarse_alignment.pkl"


# Load Source Data
with open(en_subtitle_path, 'rb') as f:
    en_subtitle = pkl.load(f)
with open(other_subtitle_path, 'rb') as f:
    other_subtitle = pkl.load(f)
with open(transcript_path, 'rb') as f:
    data = pkl.load(f)

# Fetch episode_key for search
# episode_keys = set()
# for x in data:
#     print(x)
#     episode_keys.add(x.strip().split('_')[0])
episode_keys = set(data.keys())


# Perform alignment between transcript and subtitles
episode_indexs = {}
for item in tqdm(tuple(episode_keys)):
    temp = []
    print(item)
    print(data.keys())
    segments = get_segments(item, data)
    for segment in tqdm(segments):
        for i, subtitle in enumerate(en_subtitle):
            if segment in subtitle:
                temp.append([i, segment, en_subtitle[i], other_subtitle[i]])
    episode_indexs[item] = temp

# Save alignment indexs
pkl.dump(episode_indexs, open(output_path, 'wb'))
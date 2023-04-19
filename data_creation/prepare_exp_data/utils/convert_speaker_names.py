from copy import deepcopy
import random

def build_speaker_dict(scene, all_names, male_names, female_names):
    scene_speakers = set()
    # Collect Speaker Names
    for line in scene:
        if len(line)==13:
            scene_speakers.add(line[8].lower())
        elif len(line)>13:
            scene_speakers.add(line[9].lower())
            # all_speakers.add(" ".join([token.lower() for token in line[9:len(line)-4]]))
    # Build Speaker Dict
    speaker_dict = {}
    male_subset = deepcopy(all_names['male_names'][:100])
    female_subset = deepcopy(all_names['female_names'][:100])
    # print(male_subset)
    # print(female_subset)
    for speaker in scene_speakers:
        if speaker in male_names:
            mapped_name = random.sample(male_subset, 1)[0]
            male_subset.remove(mapped_name)
            speaker_dict[speaker] = mapped_name.capitalize()
        elif speaker in female_names:
            # print("Female:", speaker)
            mapped_name = random.sample(female_subset, 1)[0]
            female_subset.remove(mapped_name)
            speaker_dict[speaker] = mapped_name.capitalize()
    return speaker_dict















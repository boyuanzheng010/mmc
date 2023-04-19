#!/bin/bash
#SBATCH -o "/home/user_id_1/hoi/pred_slurms/slurm-%j.out"
#python predict.py train_spanbert_large_ml0_d1 Jun11_21-48-45_9000 ontonotes english 3
#python predict.py train_spanbert_large_ml0_d1 Jun11_21-48-32_9000 friends english 3
#python predict.py train_spanbert_large_ml0_d1 Jun13_16-42-51_3000 dialogue english 2

#python evaluate.py train_xlmr_base_ml0_d1 Jun16_11-01-34_5000 ontonotes chinese 6
#python evaluate.py train_xlmr_base_ml0_d1 Jun16_00-58-20_7600 dialogue chinese 6
#python evaluate.py train_xlmr_base_ml0_d1 Jun16_01-33-15_21600 dialogue english 6
#python predict.py train_xlmr_base_ml0_d1 Jun16_01-33-15_21600 dialogue english 3
#python predict.py train_xlmr_base_ml0_d1 Jun20_20-36-20_15000 dialogue chinese 3
#python predict.py train_xlmr_base_ml0_d1 Jun20_20-36-39_13000 dialogue farsi 3

#python predict.py train_xlmr_base_ml0_d1 Jun16_14-34-14_33000 mix_ontonotes_dialogue chinese 1
#python predict.py train_xlmr_base_ml0_d1 Jun16_14-58-28_23700 mix_ontonotes_dialogue english 2
#python predict.py train_xlmr_base_ml0_d1 Jun16_14-54-17_27000 mix_ontonotes_dialogue english_chinese 3
#python predict.py train_xlmr_base_ml0_d1 Jun16_15-45-24_10100 continual_dialogue chinese 5
#python predict.py train_xlmr_base_ml0_d1 Jun16_15-38-49_22700 continual_dialogue english 6

#python predict.py train_xlmr_base_ml0_d1 Jun21_14-14-14_12000 dialogue chinese 7
#python predict.py train_xlmr_base_ml0_d1 Jun21_14-13-39_11000 dialogue farsi 7
#python predict.py train_xlmr_base_ml0_d1 Jun16_01-33-15_21600 dialogue english 7


# Do SpanBert
#python predict.py train_spanbert_large_ml0_d1_dialogue_english Jun22_19-11-56_best dialogue english 10
#python predict.py train_spanbert_large_ml0_d1_friends_english Jun22_19-12-03_best friends english 10
#python predict.py train_spanbert_large_ml0_d1_ontonotes_english Jun22_19-11-58_best ontonotes english 10
#python predict.py train_spanbert_large_ml0_d1_mix_english Jun22_20-26-07_best mix_ontonotes_dialogue english 10

# Chinese Side Experiments
#python predict.py train_xlmr_base_ml0_d1_ontonotes_chinese Jun23_01-22-32_best ontonotes chinese 4 #
#python predict.py train_xlmr_base_ml0_d1_dialogue_chinese Jun22_19-11-29_best dialogue chinese 0 #
#python predict.py train_xlmr_base_ml0_d1_mix_english Jun23_00-07-38_best mix_ontonotes_dialogue english 3
#python predict.py train_xlmr_base_ml0_d1_ontonotes_chinese Jun23_18-00-09_best dialogue chinese 1



#python predict.py train_xlmr_base_ml0_d1_mix_chinese Jun22_19-11-41_best mix_ontonotes_dialogue chinese 1
#python predict.py train_xlmr_base_ml0_d1_dialogue_farsi Jun22_19-11-26_best dialogue farsi 2




#python predict.py train_xlmr_base_ml0_d1_dialogue_chinese_english_farsi Jun24_01-11-58_best dialogue chinese_english_farsi 2


#python predict.py train_xlmr_base_ml0_d1_tbbt_english Jun24_01-47-56_best tbbt english 0
#python predict.py train_xlmr_base_ml0_d1_friends_english Jun24_01-48-07_best friends english 1
#python predict.py train_xlmr_base_ml0_d1_dialogue_english Jun23_23-12-22_best dialogue english 2


#python predict.py train_xlmr_base_ml0_d1_dialogue_chinese_english Jun23_13-13-23_best dialogue chinese_english 0 # Dialogue English+Chinese

# Continual Training
#python predict.py train_spanbert_large_ml0_d1_ontonotes_english Jun23_18-05-38_best dialogue english 0
#python predict.py train_xlmr_base_ml0_d1_ontonotes_chinese Jun23_18-00-09_best dialogue chinese 1


# Do OntoNotes Short, Medium, Long
#python predict.py train_spanbert_large_ml0_d1_ontonotes_english Jun22_19-11-58_best ontonotes english 10
#python predict.py train_spanbert_large_ml0_d1_friends_english Jun22_19-12-03_best friends english 1
#python predict.py train_spanbert_large_ml0_d1_dialogue_english Jun22_19-11-56_best dialogue english 2
#python predict.py train_spanbert_large_ml0_d1_mix_english Jun22_20-26-07_best mix_ontonotes_dialogue english 3
#python predict.py train_spanbert_large_ml0_d1_ontonotes_english Jun23_18-05-38_best dialogue english 4


# Chinese Side Experiments
#python predict.py train_xlmr_base_ml0_d1_ontonotes_chinese Jun23_01-22-32_best ontonotes chinese 0 #
#python predict.py train_xlmr_base_ml0_d1_dialogue_chinese Jun22_19-11-29_best dialogue chinese 1 #
#python predict.py train_xlmr_base_ml0_d1 Jun16_14-34-14_33000 mix_ontonotes_dialogue chinese 1
#python predict.py train_xlmr_base_ml0_d1_ontonotes_chinese Jun23_18-00-09_best dialogue chinese 3





### Re-Run Experiments
#python predict.py train_spanbert_large_ml0_d1_ontonotes_english Jul25_05-02-34_best ontonotes english 10 # Best Trained
#python predict.py train_spanbert_large_ml0_d1_ontonotes_english May10_03-28-49_best ontonotes english 7 # off-the-Shelf

#python predict.py train_spanbert_large_ml0_d1_mix_ontonotes_dialogue_english Jul20_15-20-47_best mix_ontonotes_dialogue english 10
#python predict.py train_spanbert_large_ml0_d1_friends_english Jul25_05-18-47_best friends english 10 # MC-Friends
#python predict.py train_spanbert_large_ml0_d1_tbbt_english Jul25_05-37-35_best tbbt english 6 # MC-TBBT

#python predict.py train_spanbert_large_ml0_d1_dialogue_english Jul19_23-14-34_best dialogue english 6

#python predict.py train_xlmr_base_ml0_d1_dialogue_chinese_english_farsi Jul25_07-36-05_best dialogue chinese_english_farsi 6

#python predict.py train_spanbert_large_ml0_d1_ontonotes_english Jul21_15-30-13_best dialogue english 10 # Continual Training in English



#python predict.py train_spanbert_large_ml0_d1_ci_english Jul25_10-35-34_best ci english 6


#python predict.py train_xlmr_base_ml0_d1_dialogue_chinese Jul24_18-33-16_best dialogue chinese 6
#python predict.py train_xlmr_base_ml0_d1_dialogue_farsi Jul25_04-04-52_best dialogue farsi 6
#python predict.py train_xlmr_base_ml0_d1_dialogue_english Jul19_23-05-47_best dialogue_xlmr english 2

#python predict.py train_spanbert_large_ml0_d1_ontonotes_english Jul25_05-02-34_best ontonotes english 6 # Best Trained



# Final Round Evaluation
#python predict.py train_spanbert_large_ml0_d1_ontonotes_english May10_03-28-49_best ontonotes english 10 # off-the-Shelf
#python predict.py train_spanbert_large_ml0_d1_ci_english Jul25_21-40-06_best ci english 10
#python predict.py train_spanbert_large_ml0_d1_dialogue_english Jul25_21-40-09_best dialogue english 10
#python predict.py train_spanbert_large_ml0_d1_mix_ontonotes_dialogue_english Jul25_21-40-09_best mix_ontonotes_dialogue english 10
#python predict.py train_spanbert_large_ml0_d1_ontonotes_english Jul25_23-33-15_best dialogue english 2 # Continual Training in English_1
#python predict.py train_spanbert_large_ml0_d1_ontonotes_english Jul25_23-34-16_best dialogue english 10 # Continual Training in English_2
#python predict.py train_spanbert_large_ml0_d1_ontonotes_english Jul26_20-56-10_best dialogue_name_replaced english 10 # Continual Training in English Name Replaced-1
#python predict.py train_spanbert_large_ml0_d1_ontonotes_english Jul26_21-45-13_best dialogue_name_replaced english 10 # Continual Training in English Name Replaced-2
#python predict.py train_spanbert_large_ml0_d1_ontonotes_english Jul26_21-45-15_best dialogue_name_replaced english 10 # Continual Training in English Name Replaced-3
#python predict.py train_spanbert_large_ml0_d1_tbbt_english Jul25_21-35-44_best tbbt english 10 # TBBT-1
#python predict.py train_spanbert_large_ml0_d1_tbbt_english Jul25_21-40-27_best tbbt english 10 # TBBT-2
#python predict.py train_spanbert_large_ml0_d1_friends_english Jul25_21-35-34_best friends english 10 # Friends-1
#python predict.py train_spanbert_large_ml0_d1_friends_english Jul25_21-40-20_best friends english 10 # Friends-2
#python predict.py train_spanbert_large_ml0_d1_dialogue_name_replaced_english Jul26_15-52-12_best dialogue_name_replaced english 10 # Name Replaced Model
#python predict.py train_spanbert_large_ml0_d1_dialogue_name_replaced_english Jul26_20-06-48_best dialogue_name_replaced english 10 # Name Replaced Model


#python predict.py train_xlmr_base_ml0_d1_ontonotes_chinese Jul25_21-36-26_best ontonotes chinese 10 # ON-1
#python predict.py train_xlmr_base_ml0_d1_ontonotes_chinese Jul25_23-32-42_best ontonotes chinese 10 # ON-2
#python predict.py train_xlmr_base_ml0_d1_dialogue_chinese Jul25_21-36-13_best dialogue chinese 10 # MC-1
#python predict.py train_xlmr_base_ml0_d1_dialogue_chinese Jul25_21-44-03_best dialogue chinese 7 # MC-2
#python predict.py train_xlmr_base_ml0_d1_mix_chinese Jul26_13-43-59_best mix_ontonotes_dialogue chinese 10 # Mix
#python predict.py train_xlmr_base_ml0_d1_ontonotes_chinese Jul25_23-34-17_best dialogue chinese 10 # Continual Training in Chinese_1
#python predict.py train_xlmr_base_ml0_d1_ontonotes_chinese Jul25_23-34-21_best dialogue chinese 10 # Continual Training in Chinese_2
#python predict.py train_xlmr_base_ml0_d1_ontonotes_chinese Jul26_15-26-57_best dialogue chinese 10 # Continual Training in Chinese_3
#python predict.py train_xlmr_base_ml0_d1_dialogue_english Jul25_21-36-13_best dialogue_xlmr english 10 # Better Results 63.15
#python predict.py train_xlmr_base_ml0_d1_dialogue_english Jul25_23-31-51_best dialogue_xlmr english 10 # Normal Results
#python predict.py train_xlmr_base_ml0_d1_dialogue_chinese Jul25_21-44-03_best dialogue chinese 10 # MC-2
#python predict.py train_xlmr_base_ml0_d1_dialogue_farsi Jul25_04-04-52_best dialogue farsi 10
#python predict.py train_xlmr_base_ml0_d1_dialogue_chinese_english_farsi Jul26_09-30-57_best dialogue chinese_english_farsi 10 # Best
#python predict.py train_xlmr_base_ml0_d1_dialogue_chinese_english_farsi Jul25_23-32-20_best dialogue chinese_english_farsi 10 # Second
#python predict.py train_xlmr_base_ml0_d1_dialogue_chinese_english_farsi Jul25_21-36-15_best dialogue chinese_english_farsi 10 # Third
#python predict.py train_xlmr_base_ml0_d1_dialogue_chinese Jul29_15-35-27_best dialogue chinese 10 # MC-2


# Evaluate-3-Random-Seeds Experiments
#python predict.py train_spanbert_large_ml0_d1_mix_ontonotes_dialogue_english Jul29_13-47-30_best mix_ontonotes_dialogue english 10
#python predict.py train_spanbert_large_ml0_d1_mix_ontonotes_dialogue_english Jul29_13-47-39_best mix_ontonotes_dialogue english 10
#python predict.py train_spanbert_large_ml0_d1_mix_ontonotes_dialogue_english Jul29_13-47-34_best mix_ontonotes_dialogue english 10
#python predict.py train_spanbert_large_ml0_d1_ontonotes_english Jul25_23-34-16_best dialogue english 10 # Continual Training in English
#python predict.py train_spanbert_large_ml0_d1_ontonotes_english Jul25_23-34-16_best dialogue english 10 # Continual Training in English
#python predict.py train_spanbert_large_ml0_d1_ontonotes_english Jul25_23-34-16_best dialogue english 10 # Continual Training in English
#python predict.py train_spanbert_large_ml0_d1_friends_english Jul29_13-47-51_best friends english 10
#python predict.py train_spanbert_large_ml0_d1_friends_english Jul29_13-47-46_best friends english 10
#python predict.py train_spanbert_large_ml0_d1_friends_english Jul29_13-47-59_best friends english 10
#python predict.py train_spanbert_large_ml0_d1_tbbt_english Jul29_13-48-04_best tbbt english 10
#python predict.py train_spanbert_large_ml0_d1_tbbt_english Jul29_13-48-11_best tbbt english 10
#python predict.py train_spanbert_large_ml0_d1_tbbt_english Jul29_13-48-15_best tbbt english 10
#python predict.py train_spanbert_large_ml0_d1_dialogue_name_replaced_english Jul29_13-47-20_best dialogue_name_replaced english 10
#python predict.py train_spanbert_large_ml0_d1_dialogue_name_replaced_english Jul29_13-47-16_best dialogue_name_replaced english 10
#python predict.py train_spanbert_large_ml0_d1_dialogue_name_replaced_english Jul29_13-47-25_best dialogue_name_replaced english 10
#python predict.py train_spanbert_large_ml0_d1_ontonotes_english Jul29_13-43-48_best ontonotes english 10
#python predict.py train_spanbert_large_ml0_d1_ontonotes_english Jul29_13-44-02_best ontonotes english 10
#python predict.py train_spanbert_large_ml0_d1_ontonotes_english Jul29_13-44-09_best ontonotes english 10
#python predict.py train_spanbert_large_ml0_d1_ci_english Jul29_13-46-40_best ci english 10
#python predict.py train_spanbert_large_ml0_d1_ci_english Jul29_13-46-46_best ci english 10
#python predict.py train_spanbert_large_ml0_d1_ci_english Jul29_13-46-54_best ci english 10
#python predict.py train_spanbert_large_ml0_d1_dialogue_english Jul29_15-28-03_best dialogue english 10
#python predict.py train_spanbert_large_ml0_d1_dialogue_english Jul29_13-47-09_best dialogue english 10
#python predict.py train_spanbert_large_ml0_d1_dialogue_english Jul29_15-28-12_best dialogue english 10



#python predict.py train_xlmr_base_ml0_d1_ontonotes_chinese Jul25_23-32-42_best ontonotes chinese 10
#python predict.py train_xlmr_base_ml0_d1_mix_chinese Jul26_13-43-59_best mix_ontonotes_dialogue chinese 10

#python predict.py train_xlmr_base_ml0_d1_dialogue_farsi Jul29_15-34-30_best dialogue farsi 1
python predict.py train_xlmr_base_ml0_d1_dialogue_farsi Jul29_15-34-39_best dialogue farsi 2
#python predict.py train_xlmr_base_ml0_d1_dialogue_farsi Jul29_15-34-44_best dialogue farsi 3
#python predict.py train_xlmr_base_ml0_d1_dialogue_chinese_english_farsi Jul29_15-34-53_best dialogue chinese_english_farsi 4
#python predict.py train_xlmr_base_ml0_d1_dialogue_chinese_english_farsi Jul29_15-35-00_best dialogue chinese_english_farsi 5
#python predict.py train_xlmr_base_ml0_d1_dialogue_chinese_english_farsi Jul29_15-34-59_best dialogue chinese_english_farsi 6
#python predict.py train_xlmr_base_ml0_d1_dialogue_english Jul29_15-35-32_best dialogue_xlmr english 1
python predict.py train_xlmr_base_ml0_d1_dialogue_english Jul29_15-35-35_best dialogue_xlmr english 2
#python predict.py train_xlmr_base_ml0_d1_dialogue_english Jul29_15-35-39_best dialogue_xlmr english 3
#python predict.py train_xlmr_base_ml0_d1_dialogue_chinese Jul29_15-35-23_best dialogue chinese 4
python predict.py train_xlmr_base_ml0_d1_dialogue_chinese Jul29_15-35-19_best dialogue chinese 2
#python predict.py train_xlmr_base_ml0_d1_dialogue_chinese Jul29_15-35-27_best dialogue chinese 6

#python predict.py train_xlmr_base_ml0_d1_ontonotes_chinese Jul31_11-13-26_best ontonotes chinese 10
#python predict.py train_xlmr_base_ml0_d1_ontonotes_chinese Jul31_11-13-35_best ontonotes chinese 10
#python predict.py train_xlmr_base_ml0_d1_ontonotes_chinese Jul31_11-13-32_best ontonotes chinese 10
#python predict.py train_xlmr_base_ml0_d1_mix_chinese Jul31_11-13-08_best mix_ontonotes_dialogue chinese 10
#python predict.py train_xlmr_base_ml0_d1_mix_chinese Jul31_11-13-13_best mix_ontonotes_dialogue chinese 10
#python predict.py train_xlmr_base_ml0_d1_mix_chinese Jul31_11-13-18_best mix_ontonotes_dialogue chinese 10
#python predict.py train_xlmr_base_ml0_d1_ontonotes_chinese Jul31_11-14-41_best dialogue chinese 10





#python predict.py train_xlmr_base_ml0_d1_golden_mention_loss_dialogue_farsi Aug15_16-18-48_best dialogue farsi 2
#python predict.py train_xlmr_base_ml0_d1_all_mention_loss_dialogue_farsi Aug15_16-18-59_best dialogue farsi 10
#python predict.py train_xlmr_base_ml0_d1_golden_mention_loss_dialogue_chinese Aug15_16-18-29_best dialogue chinese 10
#python predict.py train_xlmr_base_ml0_d1_all_mention_loss_dialogue_chinese Aug15_16-18-38_best dialogue chinese 10
#python predict.py train_spanbert_large_ml0_d1_golden_mention_loss_dialogue_english Aug15_16-18-02_best dialogue english 10
#python predict.py train_spanbert_large_ml0_d1_all_mention_loss_dialogue_english Aug15_16-18-18_best dialogue english 10

#python predict.py train_spanbert_large_ml0_d1_weighted_all_0_9_5_mention_loss_dialogue_english Aug16_16-24-44_best dialogue english 7
#python predict.py train_spanbert_large_ml0_d1_weighted_all_0_9_mention_loss_dialogue_english Aug16_16-24-50_best dialogue english 7
#python predict.py train_spanbert_large_ml0_d1_weighted_all_0_8_5_mention_loss_dialogue_english Aug16_16-25-01_best dialogue english 7
#python predict.py train_spanbert_large_ml0_d1_weighted_all_0_8_mention_loss_dialogue_english Aug16_16-25-09_best dialogue english 7
#python predict.py train_spanbert_large_ml0_d1_weighted_all_0_7_mention_loss_dialogue_english Aug16_16-25-17_best dialogue english 7
#python predict.py train_spanbert_large_ml0_d1_weighted_all_0_6_mention_loss_dialogue_english Aug16_16-25-30_best dialogue english 7
#python predict.py train_spanbert_large_ml0_d1_weighted_all_0_5_mention_loss_dialogue_english Aug16_16-25-38_best dialogue english 7
#python predict.py train_spanbert_large_ml0_d1_weighted_all_0_4_mention_loss_dialogue_english Aug16_16-25-47_best dialogue english 7
#python predict.py train_spanbert_large_ml0_d1_weighted_all_0_3_mention_loss_dialogue_english Aug16_16-25-54_best dialogue english 7
#python predict.py train_spanbert_large_ml0_d1_weighted_all_0_2_mention_loss_dialogue_english Aug16_16-26-02_best dialogue english 7
#python predict.py train_spanbert_large_ml0_d1_weighted_all_0_1_mention_loss_dialogue_english Aug16_16-26-12_best dialogue english 7

#python predict.py train_xlmr_base_ml0_d1_weighted_all_0_8_5_mention_loss_dialogue_chinese Aug16_16-41-27_best dialogue chinese 3
#python predict.py train_xlmr_base_ml0_d1_weighted_all_0_7_mention_loss_dialogue_chinese Aug16_16-42-56_best dialogue chinese 3
#python predict.py train_xlmr_base_ml0_d1_weighted_all_0_5_5_mention_loss_dialogue_chinese Aug16_16-42-56_best dialogue chinese 3

#python predict.py train_xlmr_base_ml0_d1_weighted_all_0_8_5_mention_loss_dialogue_farsi Aug16_16-43-21_best dialogue farsi 7
#python predict.py train_xlmr_base_ml0_d1_weighted_all_0_7_mention_loss_dialogue_farsi Aug16_16-43-24_best dialogue farsi 7
#python predict.py train_xlmr_base_ml0_d1_weighted_all_0_5_5_mention_loss_dialogue_farsi Aug16_16-43-24_best dialogue farsi 1

#python predict.py train_xlmr_base_ml0_d1_weighted_all_0_7_mention_loss_dialogue_english Aug29_14-58-11_best dialogue_xlmr english 6
#python predict.py train_xlmr_base_ml0_d1_golden_mention_loss_dialogue_english Aug29_19-31-29_best dialogue_xlmr english 6


#python predict.py train_xlmr_base_ml0_d1_weighted_all_0_4_mention_loss_dialogue_farsi_reduce Aug31_21-21-48_best dialogue_reduced_dev farsi 1
#python predict.py train_xlmr_base_ml0_d1_golden_mention_loss_dialogue_farsi_reduce Aug31_20-02-25_best dialogue_reduced_dev farsi 2
#python predict.py train_xlmr_base_ml0_d1_all_mention_loss_dialogue_farsi_reduce Aug31_20-05-09_best dialogue_reduced_dev farsi 3

#python predict.py train_xlmr_base_ml0_d1_dialogue_chinese_english_farsi Sep01_05-44-23_best dialogue chinese_english_farsi 0

#python predict.py train_spanbert_large_ml0_d1_weighted_all_0_3_mention_loss_dialogue_english Sep01_13-47-21_best dialogue english 6
#python predict.py train_spanbert_large_ml0_d1_weighted_all_0_4_mention_loss_dialogue_english Sep01_11-09-09_best dialogue english 6
python predict.py train_spanbert_large_ml0_d1_weighted_all_0_8_mention_loss_dialogue_english Sep01_11-09-10_best dialogue english 5


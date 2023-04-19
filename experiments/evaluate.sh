#!/bin/bash
#python evaluate.py train_spanbert_large_ml0_d1 Jun11_21-48-45_9000 ontonotes english 10
#python evaluate.py train_spanbert_large_ml0_d1 Jun11_21-48-32_9000 friends english 10
#python evaluate.py train_spanbert_large_ml0_d1 Jun11_21-48-37_6000 dialogue english 3
#python evaluate.py train_xlmr_base_ml0_d1 Jun16_11-01-34_5000 ontonotes chinese 1
#python evaluate.py train_xlmr_base_ml0_d1 Jun19_14-04-01_5000 ontonotes chinese 1 # OntoNotes Chinese(OntoNotes-zh Dev)

#python evaluate.py train_xlmr_base_ml0_d1 Jun20_20-36-39_13000 dialogue farsi 4
#python evaluate.py train_xlmr_base_ml0_d1 Jun16_01-33-15_21600 dialogue english 6
#python evaluate.py train_xlmr_base_ml0_d1 Jun16_00-58-20_7600 dialogue chinese 1
#python evaluate.py train_xlmr_base_ml0_d1 Jun16_00-58-20_7600 dialogue chinese 1

#python evaluate.py train_xlmr_base_ml0_d1 Jun16_14-34-14_33000 mix_ontonotes_dialogue chinese 1
#python evaluate.py train_xlmr_base_ml0_d1 Jun16_14-58-28_23700 mix_ontonotes_dialogue english 2
#python evaluate.py train_xlmr_base_ml0_d1 Jun16_14-54-17_27000 mix_ontonotes_dialogue english_chinese 3
#python evaluate.py train_xlmr_base_ml0_d1 Jun16_15-45-24_10100 continual_dialogue chinese 4
#python evaluate.py train_xlmr_base_ml0_d1 Jun16_15-38-49_22700 continual_dialogue english 0

# Evaluate Re-Run Models
python evaluate.py train_spanbert_large_ml0_d1_dialogue_english Jul19_23-14-34_best dialogue english 10

import json

from run import Runner
import sys
import os
import pickle as pkl



def single_predict(dataset, language, runner, model):
    print('===================='+f'{dataset+"-"+language}'+'====================')
    examples_train, examples_dev, examples_test = runner.data.get_evaluation_tensor_examples(dataset, language)
    stored_info = runner.data.get_stored_info()
    print("Dev Results:")
    dev_predictions = runner.predict(model, examples_dev, stored_info, 0, official=True)
    print("Test Results:")
    test_predictions = runner.predict(model, examples_test, stored_info, 0, official=True)
    # print("Train Results:")
    # train_predictions = runner.predict(model, examples_train, stored_info, 0, official=False)
    to_save = {
        "examples_train": examples_train,
        "examples_dev": examples_dev,
        "examples_test": examples_test,
        "stored_info": stored_info,
        # "train_predictions": train_predictions,
        "dev_predictions": dev_predictions,
        "test_predictions": test_predictions
    }
    with open(os.path.join(runner.config['root_dir']+"/predictions", dataset+"_"+language+"_trained_on_"+train_dataset+"_"+train_language+".pkl"), 'wb') as f:
        pkl.dump(to_save, f)

def predict(config_name, gpu_id, saved_suffix, train_dataset, train_language):
    runner = Runner(config_name, gpu_id)
    runner.modify_config_dataset(train_dataset)
    runner.modify_config_language(train_language)
    runner.re_initialize_config()

    model = runner.initialize_model(saved_suffix)

    """
    English Side Experiments
    """
    # single_predict("ci", "english", runner, model)
    # single_predict("dialogue", "english", runner, model)
    # single_predict("friends", "english", runner, model)
    # single_predict("tbbt", "english", runner, model)
    # single_predict("dialogue_name_replaced", "english", runner, model)
    # single_predict("friends_name_replaced", "english", runner, model)
    # single_predict("tbbt_name_replaced", "english", runner, model)

    single_predict("ontonotes", "english", runner, model)
    # single_predict("ontonotes_short", "english", runner, model)
    # single_predict("ontonotes_medium", "english", runner, model)
    # single_predict("ontonotes_long", "english", runner, model)
    # single_predict("dialogue_name_replaced", "english", runner, model)
    # single_predict("dialogue_name_replaced_2", "english", runner, model)
    # single_predict("dialogue_name_replaced_3", "english", runner, model)
    # single_predict("dialogue_xlmr", "english", runner, model)
    # single_predict("dialogue_name_replaced_xlmr", "english", runner, model)

    single_predict("dialogue_xlmr", "english", runner, model)
    # single_predict("friends_xlmr", "english", runner, model)
    # single_predict("tbbt_xlmr", "english", runner, model)
    # single_predict("ci_xlmr", "english", runner, model)
    # single_predict("ontonotes_xlmr_long", "english", runner, model)
    # single_predict("ontonotes_xlmr_medium", "english", runner, model)
    # single_predict("ontonotes_xlmr_short", "english", runner, model)
    # single_predict("ontonotes_xlmr", "english", runner, model)


    # """
    # Chinese Side Experiments
    # """
    # single_predict("ontonotes", "chinese", runner, model)
    # single_predict("ontonotes_short", "chinese", runner, model)
    # single_predict("ontonotes_medium", "chinese", runner, model)
    # single_predict("ontonotes_long", "chinese", runner, model)
    # single_predict("dialogue", "chinese", runner, model)
    single_predict("dialogue_all_corrected", "chinese", runner, model)
    # single_predict("dialogue_uncorrected", "chinese", runner, model)
    # single_predict("friends", "chinese", runner, model)
    # single_predict("tbbt", "chinese", runner, model)
    # """
    # Farsi Side Experiments
    # """
    # single_predict("ontonotes", "farsi", runner, model)
    # single_predict("ontonotes_short", "farsi", runner, model)
    # single_predict("ontonotes_medium", "farsi", runner, model)
    # single_predict("ontonotes_long", "farsi", runner, model)
    # single_predict("dialogue", "farsi", runner, model)
    # single_predict("friends", "farsi", runner, model)
    # single_predict("tbbt", "farsi", runner, model)
    single_predict("dialogue_corrected", "farsi", runner, model)
    # single_predict("dialogue_uncorrected", "farsi", runner, model)

if __name__ == '__main__':
    print("###############", sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], int(sys.argv[5]))
    config_name, saved_suffix, train_dataset, train_language, gpu_id = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], int(sys.argv[5])
    if gpu_id == 10:
        gpu_id = -1
    predict(config_name, gpu_id, saved_suffix, train_dataset, train_language)

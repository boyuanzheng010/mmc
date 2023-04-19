from run import Runner
import sys


def evaluate(config_name, gpu_id, saved_suffix, train_dataset, train_language):
    runner = Runner(config_name, gpu_id)
    runner.modify_config_dataset(train_dataset)
    runner.modify_config_language(train_language)
    runner.re_initialize_config()

    model = runner.initialize_model(saved_suffix)

    # # Evaluate on Dialogue-Chinese
    # dataset = "dialogue"
    # language = "chinese"
    # print('===================='+f'{dataset+"-"+language}'+'====================')
    # examples_train, examples_dev, examples_test = runner.data.get_evaluation_tensor_examples(dataset, language)
    # stored_info = runner.data.get_stored_info()
    # print("Dev Results:")
    # runner.decoupled_evaluate(model, examples_dev, stored_info, 0, official=False)
    # print("Test Results:")
    # runner.decoupled_evaluate(model, examples_test, stored_info, 0, official=False)
    # print("Train Results:")
    # runner.decoupled_evaluate(model, examples_train, stored_info, 0, official=False)

    # Evaluate on Dialogue-English
    dataset = "dialogue"
    language = "english"
    print('===================='+f'{dataset+"-"+language}'+'====================')
    examples_train, examples_dev, examples_test = runner.data.get_evaluation_tensor_examples(dataset, language)
    stored_info = runner.data.get_stored_info()
    print("Dev Results:")
    runner.decoupled_evaluate(model, examples_dev, stored_info, 0, official=False)
    print("Test Results:")
    runner.decoupled_evaluate(model, examples_test, stored_info, 0, official=False)
    print("Train Results:")
    runner.decoupled_evaluate(model, examples_train, stored_info, 0, official=False)

    # # Evaluate on Dialogue-Farsi
    # dataset = "dialogue"
    # language = "farsi"
    # print('===================='+f'{dataset+"-"+language}'+'====================')
    # examples_train, examples_dev, examples_test = runner.data.get_evaluation_tensor_examples(dataset, language)
    # stored_info = runner.data.get_stored_info()
    # print("Dev Results:")
    # runner.decoupled_evaluate(model, examples_dev, stored_info, 0, official=False)
    # print("Test Results:")
    # runner.decoupled_evaluate(model, examples_test, stored_info, 0, official=False)
    # print("Train Results:")
    # runner.decoupled_evaluate(model, examples_train, stored_info, 0, official=False)

    # # Evaluate on OntoNotes-Chinese
    # dataset = "ontonotes"
    # language = "chinese"
    # print('===================='+f'{dataset+"-"+language}'+'====================')
    # examples_train, examples_dev, examples_test = runner.data.get_evaluation_tensor_examples(dataset, language)
    # stored_info = runner.data.get_stored_info()
    # print("Dev Results:")
    # runner.decoupled_evaluate(model, examples_dev, stored_info, 0, official=False)
    # print("Test Results:")
    # runner.decoupled_evaluate(model, examples_test, stored_info, 0, official=False)
    # print("Train Results:")
    # runner.decoupled_evaluate(model, examples_train, stored_info, 0, official=False)


    # Evaluate on OntoNotes-English
    dataset = "ontonotes"
    language = "english"
    print('===================='+f'{dataset+"-"+language}'+'====================')
    examples_train, examples_dev, examples_test = runner.data.get_evaluation_tensor_examples(dataset, language)
    stored_info = runner.data.get_stored_info()
    print("Dev Results:")
    runner.decoupled_evaluate(model, examples_dev, stored_info, 0, official=False)
    print("Test Results:")
    runner.decoupled_evaluate(model, examples_test, stored_info, 0, official=False)
    print("Train Results:")
    runner.decoupled_evaluate(model, examples_train, stored_info, 0, official=False)

    # Evaluate on Friends-English
    dataset = "friends"
    language = "english"
    print('===================='+f'{dataset+"-"+language}'+'====================')
    examples_train, examples_dev, examples_test = runner.data.get_evaluation_tensor_examples(dataset, language)
    stored_info = runner.data.get_stored_info()
    print("Dev Results:")
    runner.decoupled_evaluate(model, examples_dev, stored_info, 0, official=False)
    print("Test Results:")
    runner.decoupled_evaluate(model, examples_test, stored_info, 0, official=False)
    print("Train Results:")
    runner.decoupled_evaluate(model, examples_train, stored_info, 0, official=False)

    # # print("Dev Set Result:")
    # # # runner.evaluate(model, examples_dev, stored_info, 0, official=False, conll_path=runner.config['conll_eval_path'])  # Eval Dev
    # # runner.evaluate(model, examples_dev, stored_info, 0, official=False)  # Eval Dev
    #
    # print("Test Set Result:")
    # # runner.evaluate(model, examples_test, stored_info, 0, official=False, conll_path=runner.config['conll_test_path'])  # Eval Test
    # runner.evaluate(model, examples_test, stored_info, 0, official=False)  # Eval Test
    #
    # # runner.evaluate(model, examples_dev, stored_info, 0, official=True, conll_path=runner.config['conll_eval_path'])  # Eval dev
    # # print('=================================')
    # print("Train Set Result:")
    # # runner.evaluate(model, examples_train, stored_info, 0, official=False, conll_path=runner.config['conll_train_path'])  # Eval Train
    # runner.evaluate(model, examples_train, stored_info, 0, official=False)  # Eval Train



if __name__ == '__main__':
    print("###############", sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], int(sys.argv[5]))
    config_name, saved_suffix, train_dataset, train_language, gpu_id = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], int(sys.argv[5])
    evaluate(config_name, gpu_id, saved_suffix, train_dataset, train_language)

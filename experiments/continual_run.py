import logging
import random
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
from transformers import AdamW
from torch.optim import Adam
from tensorize import CorefDataProcessor
import util
import time
from os.path import join
from metrics import CorefEvaluator, MentionScorer
from datetime import datetime
from torch.optim.lr_scheduler import LambdaLR
from model import CorefModel
import conll
import sys
from os import makedirs
from os.path import join
import pyhocon

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger()


class Runner:
    def __init__(self, config_name, gpu_id=0, seed=None):
        self.name = config_name
        self.name_suffix = datetime.now().strftime('%b%d_%H-%M-%S')
        self.gpu_id = gpu_id
        self.seed = seed


        # Set up config
        self.config = util.initialize_config(config_name, continual=True)

        # Set up logger
        log_path = join(self.config['log_dir'], 'log_' + self.name_suffix + '.txt')
        logger.addHandler(logging.FileHandler(log_path, 'a'))
        logger.info('Log file path: %s' % log_path)

        # Config save model or not
        self.save_checkpoint = self.config['save_checkpoint']

        # Set up seed
        if seed:
            util.set_seed(seed)

        # Set up device
        if gpu_id is None:
            self.device = 'cpu'
        elif gpu_id==-1:
            self.device = 'cuda'
        else:
            self.device = f'cuda:{gpu_id}'
        # self.device = torch.device('cpu' if gpu_id is None else f'cuda:{gpu_id}')
        print("################Device:", self.device)

        # Set up data
        self.data = CorefDataProcessor(self.config)

    def modify_config_dataset(self, new_dataset):
        temp = self.config['dataset']
        self.config['dataset'] = new_dataset
        return temp

    def modify_config_language(self, new_language):
        temp = self.config['language']
        self.config['language'] = new_language
        return temp

    def re_initialize_config(self):
        self.config['log_dir'] = join(self.config["log_root"], self.config["dataset"] + "_" + self.config["language"] + "_" + self.name)
        makedirs(self.config['log_dir'], exist_ok=True)

        self.config['tb_dir'] = join(self.config['log_root'], 'tensorboard')
        makedirs(self.config['tb_dir'], exist_ok=True)
        logger.info(pyhocon.HOCONConverter.convert(self.config, "hocon"))

    def initialize_model(self, saved_suffix=None):
        model = CorefModel(self.config, self.device)
        if saved_suffix:
            self.load_model_checkpoint(model, saved_suffix)
        return model

    def train(self, model):
        conf = self.config
        logger.info(conf)
        epochs, grad_accum = conf['num_epochs'], conf['gradient_accumulation_steps']

        model.to(self.device)
        logger.info('Model parameters:')
        for name, param in model.named_parameters():
            logger.info('%s: %s' % (name, tuple(param.shape)))

        # Set up tensorboard
        tb_path = join(conf['tb_dir'], conf['dataset'] + '_' + conf['language'] + '_' + self.name + '_' + self.name_suffix)
        tb_writer = SummaryWriter(tb_path, flush_secs=30)
        logger.info('Tensorboard summary path: %s' % tb_path)

        # Set up data
        examples_train, examples_dev, examples_test = self.data.get_tensor_examples()
        stored_info = self.data.get_stored_info()

        # Set up optimizer and scheduler
        total_update_steps = len(examples_train) * epochs // grad_accum
        optimizers = self.get_optimizer(model)
        schedulers = self.get_scheduler(optimizers, total_update_steps)

        # Get model parameters for grad clipping
        # Need to modify this to adapt to Chinese and Farsi
        bert_param, task_param = model.get_params()

        # Start training
        logger.info('*******************Training*******************')
        logger.info('Num samples: %d' % len(examples_train))
        logger.info('Num epochs: %d' % epochs)
        logger.info('Gradient accumulation steps: %d' % grad_accum)
        logger.info('Total update steps: %d' % total_update_steps)

        loss_during_accum = []  # To compute effective loss at each update
        loss_during_report = 0.0  # Effective loss during logging step
        loss_history = []  # Full history of effective loss; length equals total update steps
        max_f1 = 0
        start_time = time.time()
        model.zero_grad()
        for epo in range(epochs):
            random.shuffle(examples_train)  # Shuffle training set
            for doc_key, example in examples_train:
                # Forward pass
                model.train()
                example_gpu = [d.to(self.device) for d in example]
                # try:
                #     _, loss = model(*example_gpu)
                # except:
                #     print("Wrong")
                _, loss = model(*example_gpu)

                # Backward; accumulate gradients and clip by grad norm
                if grad_accum > 1:
                    loss /= grad_accum
                loss.backward()
                if conf['max_grad_norm']:
                    torch.nn.utils.clip_grad_norm_(bert_param, conf['max_grad_norm'])
                    torch.nn.utils.clip_grad_norm_(task_param, conf['max_grad_norm'])
                loss_during_accum.append(loss.item())

                # Update
                if len(loss_during_accum) % grad_accum == 0:
                    for optimizer in optimizers:
                        optimizer.step()
                    model.zero_grad()
                    for scheduler in schedulers:
                        scheduler.step()

                    # Compute effective loss
                    effective_loss = np.sum(loss_during_accum).item()
                    loss_during_accum = []
                    loss_during_report += effective_loss
                    loss_history.append(effective_loss)

                    # Report
                    if len(loss_history) % conf['report_frequency'] == 0:
                        # Show avg loss during last report interval
                        avg_loss = loss_during_report / conf['report_frequency']
                        loss_during_report = 0.0
                        end_time = time.time()
                        logger.info('Step %d: avg loss %.2f; steps/sec %.2f' %
                                    (len(loss_history), avg_loss, conf['report_frequency'] / (end_time - start_time)))
                        start_time = end_time

                        tb_writer.add_scalar('Training_Loss', avg_loss, len(loss_history))
                        tb_writer.add_scalar('Learning_Rate_Bert', schedulers[0].get_last_lr()[0], len(loss_history))
                        tb_writer.add_scalar('Learning_Rate_Task', schedulers[1].get_last_lr()[-1], len(loss_history))

                    # Evaluate
                    if len(loss_history) > 0 and len(loss_history) % conf['eval_frequency'] == 0:
                        print("Dev Scores:")
                        # f1, _ = self.evaluate(model, examples_dev, stored_info, len(loss_history), official=False, conll_path=self.config['conll_eval_path'], tb_writer=tb_writer)
                        f1, _ = self.evaluate(model, examples_dev, stored_info, len(loss_history), official=False, tb_writer=tb_writer)
                        if f1 > max_f1:
                            max_f1 = f1
                            if self.save_checkpoint:
                                self.save_model_checkpoint(model, len(loss_history))
                        logger.info('Eval max f1: %.2f' % max_f1)
                        start_time = time.time()
                    # # Test
                    # if len(loss_history) > 0 and len(loss_history) % conf['eval_frequency'] == 0:
                    #     print("Test Scores:")
                    #     f1, _ = self.evaluate(model, examples_test, stored_info, len(loss_history), official=False, conll_path=self.config['conll_test_path'], tb_writer=tb_writer)
                    #     start_time = time.time()
                    # # Train
                    # if len(loss_history) > 0 and len(loss_history) % conf['eval_frequency'] == 0:
                    #     print("Train Scores:")
                    #     f1, _ = self.evaluate(model, examples_train, stored_info, len(loss_history), official=False, conll_path=self.config['conll_train_path'], tb_writer=tb_writer)
                    #     start_time = time.time()

        logger.info('**********Finished training**********')
        logger.info('Actual update steps: %d' % len(loss_history))

        # Wrap up
        tb_writer.close()
        return loss_history


    def my_predict(self, model, tensor_examples, stored_info, step, official=False, conll_path=None, tb_writer=None):
        logger.info('Step %d: evaluating on %d samples...' % (step, len(tensor_examples)))
        model.to(self.device)
        evaluator = CorefEvaluator()
        doc_to_prediction = {}

        model.eval()
        for i, (doc_key, tensor_example) in enumerate(tensor_examples):
            gold_clusters = stored_info['gold'][doc_key]
            tensor_example = tensor_example[:7]  # Strip out gold
            example_gpu = [d.to(self.device) for d in tensor_example]
            with torch.no_grad():
                try:
                    _, _, _, span_starts, span_ends, antecedent_idx, antecedent_scores = model(*example_gpu)
                except:
                    continue
            print("Index:", i)
            print("span_start:", span_starts)
            print("span_ends:", span_ends)
            print("antecedent_idx:", antecedent_idx)
            print("antecedent_scores:", antecedent_scores)
            print("==" * 50)
            span_starts, span_ends = span_starts.tolist(), span_ends.tolist()
            antecedent_idx, antecedent_scores = antecedent_idx.tolist(), antecedent_scores.tolist()
            predicted_clusters = model.update_evaluator(span_starts, span_ends, antecedent_idx, antecedent_scores, gold_clusters, evaluator)
            doc_to_prediction[doc_key] = predicted_clusters

            return {
                "span_starts": span_starts,
                "span_ends": span_ends,
                "antecedent_idx": antecedent_idx,
                "antecedent_scores": antecedent_scores,
                "predicted_clusters": predicted_clusters,
                "doc_to_prediction": doc_to_prediction
            }

    def decoupled_evaluate(self, model, tensor_examples, stored_info, step, official=False, conll_path=None, tb_writer=None):
        # logger.info('Step %d: evaluating on %d samples...' % (step, len(tensor_examples)))
        model.to(self.device)
        evaluator = CorefEvaluator()
        doc_to_prediction = {}

        model.eval()
        for i, (doc_key, tensor_example) in enumerate(tensor_examples):
            gold_clusters = stored_info['gold'][doc_key]
            tensor_example = tensor_example[:7]  # Strip out gold
            example_gpu = [d.to(self.device) for d in tensor_example]
            with torch.no_grad():
                try:
                    _, _, _, span_starts, span_ends, antecedent_idx, antecedent_scores, span_scores = model(
                        *example_gpu)
                except:
                    print(i, doc_key)
                    continue
                # _, _, _, span_starts, span_ends, antecedent_idx, antecedent_scores, span_scores = model(*example_gpu)
            span_starts, span_ends = span_starts.tolist(), span_ends.tolist()
            antecedent_idx, antecedent_scores = antecedent_idx.tolist(), antecedent_scores.tolist()
            predicted_clusters = model.update_evaluator(span_starts, span_ends, antecedent_idx, antecedent_scores, gold_clusters, span_scores, evaluator)
            doc_to_prediction[doc_key] = predicted_clusters

        print('%s  %s  %s  %s' % (" ".ljust(11, " "), "R".ljust(5, " "), "P".ljust(5, " "), "F".ljust(5, " ")))
        # Get Macro Score
        metric_name = "macro"
        p, r, f = evaluator.get_prf(metric_name)
        # print(metric_name.ljust(10, " "), 'Precision:', int(p*100), 'Recall:', int(r*100), 'F1:', int(f*100))
        # logger.info('%s:  %.2f  %.2f  %.2f' % (metric_name.ljust(10, " "), p * 100, r * 100, f * 100))
        print('%s:  %.2f  %.2f  %.2f' % (metric_name.ljust(10, " "), p * 100, r * 100, f * 100))

        metric_name = "muc"
        p, r, f = evaluator.get_prf(metric_name)
        # print(metric_name.ljust(10, " "), 'Precision:', int(p*100), 'Recall:', int(r*100), 'F1:', int(f*100))
        print('%s:  %.2f  %.2f  %.2f' % (metric_name.ljust(10, " "), p * 100, r * 100, f * 100))

        metric_name = "b_cubed"
        p, r, f = evaluator.get_prf(metric_name)
        # print(metric_name.ljust(10, " "), 'Precision:', int(p*100), 'Recall:', int(r*100), 'F1:', int(f*100))
        print('%s:  %.2f  %.2f  %.2f' % (metric_name.ljust(10, " "), p * 100, r * 100, f * 100))

        metric_name = "ceafe"
        p, r, f = evaluator.get_prf(metric_name)
        # print(metric_name.ljust(10, " "), 'Precision:', int(p*100), 'Recall:', int(r*100), 'F1:', int(f*100))
        print('%s:  %.2f  %.2f  %.2f' % (metric_name.ljust(10, " "), p * 100, r * 100, f * 100))

        # p, r, f = evaluator.get_prf()
        # metrics = {'Eval_Avg_Precision': p * 100, 'Eval_Avg_Recall': r * 100, 'Eval_Avg_F1': f * 100}
        # for name, score in metrics.items():
        #     logger.info('%s: %.2f' % (name, score))
        #     if tb_writer:
        #         tb_writer.add_scalar(name, score, step)

        if official:
            conll_results = conll.evaluate_conll(conll_path, doc_to_prediction, stored_info['subtoken_maps'])
            official_f1 = sum(results["f"] for results in conll_results.values()) / len(conll_results)
            logger.info('Official avg F1: %.4f' % official_f1)

        return f * 100


    def predict(self, model, tensor_examples, stored_info, step, official=False, conll_path=None, tb_writer=None):
        model.to(self.device)
        evaluator = CorefEvaluator()
        doc_to_prediction = {}

        model.eval()
        for i, (doc_key, tensor_example) in enumerate(tensor_examples):
            gold_clusters = stored_info['gold'][doc_key]
            tensor_example = tensor_example[:7]  # Strip out gold
            example_gpu = [d.to(self.device) for d in tensor_example]
            with torch.no_grad():
                _, _, _, span_starts, span_ends, antecedent_idx, antecedent_scores, span_scores = model(*example_gpu)
            span_starts, span_ends = span_starts.tolist(), span_ends.tolist()
            antecedent_idx, antecedent_scores = antecedent_idx.tolist(), antecedent_scores.tolist()
            # predicted_clusters = model.update_evaluator(span_starts, span_ends, antecedent_idx, antecedent_scores, gold_clusters, evaluator)
            predicted_clusters, predictions = model.update_evaluator_prediction(span_starts, span_ends, antecedent_idx, antecedent_scores, gold_clusters, span_scores, evaluator)
            doc_to_prediction[doc_key] = predictions




        print('%s  %s  %s  %s' % (" ".ljust(11, " "), "R".ljust(5, " "), "P".ljust(5, " "), "F".ljust(5, " ")))
        # Get Macro Score
        metric_name = "macro"
        p, r, f = evaluator.get_prf(metric_name)
        # print(metric_name.ljust(10, " "), 'Precision:', int(p*100), 'Recall:', int(r*100), 'F1:', int(f*100))
        # logger.info('%s:  %.2f  %.2f  %.2f' % (metric_name.ljust(10, " "), p * 100, r * 100, f * 100))
        print('%s:  %.2f  %.2f  %.2f' % (metric_name.ljust(10, " "), p * 100, r * 100, f * 100))

        metric_name = "muc"
        p, r, f = evaluator.get_prf(metric_name)
        # print(metric_name.ljust(10, " "), 'Precision:', int(p*100), 'Recall:', int(r*100), 'F1:', int(f*100))
        print('%s:  %.2f  %.2f  %.2f' % (metric_name.ljust(10, " "), p * 100, r * 100, f * 100))

        metric_name = "b_cubed"
        p, r, f = evaluator.get_prf(metric_name)
        # print(metric_name.ljust(10, " "), 'Precision:', int(p*100), 'Recall:', int(r*100), 'F1:', int(f*100))
        print('%s:  %.2f  %.2f  %.2f' % (metric_name.ljust(10, " "), p * 100, r * 100, f * 100))

        metric_name = "ceafe"
        p, r, f = evaluator.get_prf(metric_name)
        # print(metric_name.ljust(10, " "), 'Precision:', int(p*100), 'Recall:', int(r*100), 'F1:', int(f*100))
        print('%s:  %.2f  %.2f  %.2f' % (metric_name.ljust(10, " "), p * 100, r * 100, f * 100))

        # p, r, f = evaluator.get_prf()
        # metrics = {'Eval_Avg_Precision': p * 100, 'Eval_Avg_Recall': r * 100, 'Eval_Avg_F1': f * 100}
        # for name, score in metrics.items():
        #     logger.info('%s: %.2f' % (name, score))
        #     if tb_writer:
        #         tb_writer.add_scalar(name, score, step)

        if official:
            conll_results = conll.evaluate_conll(conll_path, doc_to_prediction, stored_info['subtoken_maps'])
            official_f1 = sum(results["f"] for results in conll_results.values()) / len(conll_results)
            logger.info('Official avg F1: %.4f' % official_f1)

        return doc_to_prediction


    # def evaluate(self, model, tensor_examples, stored_info, step, official=False, conll_path=None, tb_writer=None):
    #     logger.info('Step %d: evaluating on %d samples...' % (step, len(tensor_examples)))
    #     model.to(self.device)
    #     evaluator = CorefEvaluator()
    #     doc_to_prediction = {}
    #
    #     model.eval()
    #     for i, (doc_key, tensor_example) in enumerate(tensor_examples):
    #         gold_clusters = stored_info['gold'][doc_key]
    #         tensor_example = tensor_example[:7]  # Strip out gold
    #         example_gpu = [d.to(self.device) for d in tensor_example]
    #         with torch.no_grad():
    #             _, _, _, span_starts, span_ends, antecedent_idx, antecedent_scores, span_scores = model(*example_gpu)
    #             # try:
    #             #     _, _, _, span_starts, span_ends, antecedent_idx, antecedent_scores = model(*example_gpu)
    #             # except:
    #             #     print(example_gpu)
    #         # print("Index:", i)
    #         # print("span_start:", span_starts)
    #         # print("span_ends:", span_ends)
    #         # print("antecedent_idx:", antecedent_idx)
    #         # print("antecedent_scores:", antecedent_scores)
    #         # print("==" * 50)
    #         span_starts, span_ends = span_starts.tolist(), span_ends.tolist()
    #         antecedent_idx, antecedent_scores = antecedent_idx.tolist(), antecedent_scores.tolist()
    #         predicted_clusters = model.update_evaluator(span_starts, span_ends, antecedent_idx, antecedent_scores, gold_clusters, span_scores, evaluator)
    #         doc_to_prediction[doc_key] = predicted_clusters
    #
    #     p, r, f = evaluator.get_prf("macro")
    #     metrics = {'Eval_Avg_Precision': p * 100, 'Eval_Avg_Recall': r * 100, 'Eval_Avg_F1': f * 100}
    #     for name, score in metrics.items():
    #         logger.info('%s: %.2f' % (name, score))
    #         if tb_writer:
    #             tb_writer.add_scalar(name, score, step)
    #
    #     if official:
    #         conll_results = conll.evaluate_conll(conll_path, doc_to_prediction, stored_info['subtoken_maps'])
    #         official_f1 = sum(results["f"] for results in conll_results.values()) / len(conll_results)
    #         logger.info('Official avg F1: %.4f' % official_f1)
    #
    #     return f * 100, metrics


    def evaluate(self, model, tensor_examples, stored_info, step, official=False, conll_path=None, tb_writer=None):
        logger.info('Step %d: evaluating on %d samples...' % (step, len(tensor_examples)))
        model.to(self.device)
        evaluator = CorefEvaluator()
        mention_evaluator = MentionScorer()
        doc_to_prediction = {}

        model.eval()
        for i, (doc_key, tensor_example) in enumerate(tensor_examples):
            gold_clusters = stored_info['gold'][doc_key]
            give_golden_mention = True
            if give_golden_mention:
                tensor_example = tensor_example
            else:
                tensor_example = tensor_example[:7]  # Strip out gold
            # print(tensor_example)
            example_gpu = [d.to(self.device) for d in tensor_example]
            with torch.no_grad():
                # print(example_gpu)
                try:
                    _, _, _, span_starts, span_ends, antecedent_idx, antecedent_scores, span_scores = model(
                        *example_gpu, False)
                    span_starts, span_ends = span_starts.tolist(), span_ends.tolist()
                    antecedent_idx, antecedent_scores = antecedent_idx.tolist(), antecedent_scores.tolist()
                    predicted_clusters = model.update_evaluator(span_starts, span_ends, antecedent_idx, antecedent_scores,
                                                            gold_clusters, span_scores, evaluator, mention_evaluator)
                    doc_to_prediction[doc_key] = predicted_clusters
                except:
                    print(doc_key)
                    # print(example_gpu)
            #     _, _, _, span_starts, span_ends, antecedent_idx, antecedent_scores, span_scores = model(*example_gpu, False)
            # span_starts, span_ends = span_starts.tolist(), span_ends.tolist()
            # antecedent_idx, antecedent_scores = antecedent_idx.tolist(), antecedent_scores.tolist()
            # predicted_clusters = model.update_evaluator(span_starts, span_ends, antecedent_idx, antecedent_scores, gold_clusters, span_scores, evaluator, mention_evaluator)
            # doc_to_prediction[doc_key] = predicted_clusters

        # model.eval()
        # for i, (doc_key, tensor_example) in enumerate(tensor_examples):
        #     try:
        #         gold_clusters = stored_info['gold'][doc_key]
        #         tensor_example = tensor_example[:7]  # Strip out gold
        #         example_gpu = [d.to(self.device) for d in tensor_example]
        #         with torch.no_grad():
        #             _, _, _, span_starts, span_ends, antecedent_idx, antecedent_scores, span_scores = model(*example_gpu)
        #         span_starts, span_ends = span_starts.tolist(), span_ends.tolist()
        #         antecedent_idx, antecedent_scores = antecedent_idx.tolist(), antecedent_scores.tolist()
        #         predicted_clusters = model.update_evaluator(span_starts, span_ends, antecedent_idx, antecedent_scores, gold_clusters, span_scores, evaluator)
        #         doc_to_prediction[doc_key] = predicted_clusters
        #         # print("Predicted Cluster:")
        #         # print(predicted_clusters)
        #         # print("Golden Cluster:")
        #         # print(gold_clusters)
        #         # print()
        #     except:
        #         print(doc_key)


        # model.eval()
        # for i, (doc_key, tensor_example) in enumerate(tensor_examples):
        #     gold_clusters = stored_info['gold'][doc_key]
        #     tensor_example = tensor_example[:7]  # Strip out gold
        #     example_gpu = [d.to(self.device) for d in tensor_example]
        #     with torch.no_grad():
        #         # _, _, _, span_starts, span_ends, antecedent_idx, antecedent_scores, span_scores = model(*example_gpu)
        #         try:
        #             _, _, _, span_starts, span_ends, antecedent_idx, antecedent_scores = model(*example_gpu)
        #         except:
        #             print(example_gpu)
        #     # print("Index:", i)
        #     # print("span_start:", span_starts)
        #     # print("span_ends:", span_ends)
        #     # print("antecedent_idx:", antecedent_idx)
        #     # print("antecedent_scores:", antecedent_scores)
        #     # print("==" * 50)
        #     span_starts, span_ends = span_starts.tolist(), span_ends.tolist()
        #     antecedent_idx, antecedent_scores = antecedent_idx.tolist(), antecedent_scores.tolist()
        #     predicted_clusters = model.update_evaluator(span_starts, span_ends, antecedent_idx, antecedent_scores, gold_clusters, span_scores, evaluator)
        #     doc_to_prediction[doc_key] = predicted_clusters

        p, r, f = mention_evaluator.get_metric()
        metrics = {'Mention_Precision': p * 100, 'Mention_Recall': r * 100, 'Mention_F1': f * 100}
        for name, score in metrics.items():
            logger.info('%s: %.2f' % (name, score))
            if tb_writer:
                tb_writer.add_scalar(name, score, step)

        p, r, f = evaluator.get_prf("macro")
        metrics = {'Eval_Avg_Precision': p * 100, 'Eval_Avg_Recall': r * 100, 'Eval_Avg_F1': f * 100}
        for name, score in metrics.items():
            logger.info('%s: %.2f' % (name, score))
            if tb_writer:
                tb_writer.add_scalar(name, score, step)

        if official:
            conll_results = conll.evaluate_conll(conll_path, doc_to_prediction, stored_info['subtoken_maps'])
            official_f1 = sum(results["f"] for results in conll_results.values()) / len(conll_results)
            logger.info('Official avg F1: %.4f' % official_f1)

        return f * 100, metrics


    # def predict(self, model, tensor_examples):
    #     logger.info('Predicting %d samples...' % len(tensor_examples))
    #     model.to(self.device)
    #     predicted_spans, predicted_antecedents, predicted_clusters = [], [], []
    #
    #     model.eval()
    #     for i, (doc_key, tensor_example) in enumerate(tensor_examples):
    #         tensor_example = tensor_example[:7]
    #         example_gpu = [d.to(self.device) for d in tensor_example]
    #         with torch.no_grad():
    #             _, _, _, span_starts, span_ends, antecedent_idx, antecedent_scores = model(*example_gpu)
    #         span_starts, span_ends = span_starts.tolist(), span_ends.tolist()
    #         antecedent_idx, antecedent_scores = antecedent_idx.tolist(), antecedent_scores.tolist()
    #         clusters, mention_to_cluster_id, antecedents = model.get_predicted_clusters(span_starts, span_ends, antecedent_idx, antecedent_scores)
    #
    #         spans = [(span_start, span_end) for span_start, span_end in zip(span_starts, span_ends)]
    #         predicted_spans.append(spans)
    #         predicted_antecedents.append(antecedents)
    #         predicted_clusters.append(clusters)
    #
    #     return predicted_clusters, predicted_spans, predicted_antecedents

    def get_optimizer(self, model):
        no_decay = ['bias', 'LayerNorm.weight']
        bert_param, task_param = model.get_params(named=True)
        grouped_bert_param = [
            {
                'params': [p for n, p in bert_param if not any(nd in n for nd in no_decay)],
                'lr': self.config['bert_learning_rate'],
                'weight_decay': self.config['adam_weight_decay']
            }, {
                'params': [p for n, p in bert_param if any(nd in n for nd in no_decay)],
                'lr': self.config['bert_learning_rate'],
                'weight_decay': 0.0
            }
        ]
        optimizers = [
            AdamW(grouped_bert_param, lr=self.config['bert_learning_rate'], eps=self.config['adam_eps']),
            Adam(model.get_params()[1], lr=self.config['task_learning_rate'], eps=self.config['adam_eps'], weight_decay=0)
        ]
        return optimizers
        # grouped_parameters = [
        #     {
        #         'params': [p for n, p in bert_param if not any(nd in n for nd in no_decay)],
        #         'lr': self.config['bert_learning_rate'],
        #         'weight_decay': self.config['adam_weight_decay']
        #     }, {
        #         'params': [p for n, p in bert_param if any(nd in n for nd in no_decay)],
        #         'lr': self.config['bert_learning_rate'],
        #         'weight_decay': 0.0
        #     }, {
        #         'params': [p for n, p in task_param if not any(nd in n for nd in no_decay)],
        #         'lr': self.config['task_learning_rate'],
        #         'weight_decay': self.config['adam_weight_decay']
        #     }, {
        #         'params': [p for n, p in task_param if any(nd in n for nd in no_decay)],
        #         'lr': self.config['task_learning_rate'],
        #         'weight_decay': 0.0
        #     }
        # ]
        # optimizer = AdamW(grouped_parameters, lr=self.config['task_learning_rate'], eps=self.config['adam_eps'])
        # return optimizer

    def get_scheduler(self, optimizers, total_update_steps):
        # Only warm up bert lr
        warmup_steps = int(total_update_steps * self.config['warmup_ratio'])

        def lr_lambda_bert(current_step):
            if current_step < warmup_steps:
                return float(current_step) / float(max(1, warmup_steps))
            return max(
                0.0, float(total_update_steps - current_step) / float(max(1, total_update_steps - warmup_steps))
            )

        def lr_lambda_task(current_step):
            return max(0.0, float(total_update_steps - current_step) / float(max(1, total_update_steps)))

        schedulers = [
            LambdaLR(optimizers[0], lr_lambda_bert),
            LambdaLR(optimizers[1], lr_lambda_task)
        ]
        return schedulers
        # return LambdaLR(optimizer, [lr_lambda_bert, lr_lambda_bert, lr_lambda_task, lr_lambda_task])

    def save_model_checkpoint(self, model, step):
        # if step < 30000:
        #     return  # Debug
        path_ckpt = join(self.config['continual_save_dir'], f'model_{self.name_suffix}_{step}.bin')
        torch.save(model.state_dict(), path_ckpt)
        logger.info('Saved model to %s' % path_ckpt)

    def load_model_checkpoint(self, model, suffix):
        path_ckpt = join(self.config['log_dir'], f'model_{suffix}.bin')
        model.load_state_dict(torch.load(path_ckpt, map_location=torch.device('cpu')), strict=False)
        logger.info('Loaded model from %s' % path_ckpt)
    # def load_model_checkpoint(self, model, suffix):
    #     path_ckpt = join(self.config['evaluation_model_path'], f'model_{suffix}.bin')
    #     model.load_state_dict(torch.load(path_ckpt, map_location=torch.device('cpu')), strict=False)
    #     logger.info('Loaded model from %s' % path_ckpt)


if __name__ == '__main__':
    print("###############", sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], int(sys.argv[5]))
    config_name, saved_suffix, old_train_dataset, old_train_language, gpu_id = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], int(sys.argv[5])

    runner = Runner(config_name, gpu_id)
    new_train_dataset = runner.modify_config_dataset(old_train_dataset)
    new_train_language = runner.modify_config_language(old_train_language)
    runner.re_initialize_config()
    model = runner.initialize_model(saved_suffix)

    runner.modify_config_dataset(new_train_dataset)
    runner.modify_config_language(new_train_language)
    runner.train(model)

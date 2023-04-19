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
from transformers import AutoTokenizer
from nltk.stem import WordNetLemmatizer
from hazm import *
from tqdm import tqdm

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
        self.config = util.initialize_config(config_name)

        # Set up logger
        log_path = join(self.config['log_dir'], 'log_' + self.name_suffix + '.txt')
        logger.addHandler(logging.FileHandler(log_path, 'a'))
        logger.info('Log file path: %s' % log_path)

        # Config save model or not
        self.save_checkpoint = self.config['save_checkpoint']

        # Set up seed
        if seed:
            util.set_seed(seed)

        #Set up device
        if gpu_id is None:
            self.device = 'cpu'
        elif gpu_id==-1:
            self.device = 'cuda'
        else:
            self.device = f'cuda:{gpu_id}'
        # self.device = 'cuda'
        # self.device = torch.device('cpu' if gpu_id is None else f'cuda:{gpu_id}')
        print("################Device:", self.device)

        # Set up data
        self.data = CorefDataProcessor(self.config)

    def modify_config_dataset(self, new_dataset):
        self.config['dataset'] = new_dataset

    def modify_config_language(self, new_language):
        self.config['language'] = new_language

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
        # projection_probability = self.data.get_projection_probability(examples_train, examples_dev, examples_test)


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
                try:
                    if example[-3].shape[0] == 0:
                        continue
                    # print(example[-3].shape[0])
                    # Forward pass
                    model.train()
                    example_gpu = [d.to(self.device) for d in example]
                    # try:
                    #     _, loss = model(*example_gpu)
                    # except:
                    #     print("Wrong")
                    _, loss = model(*example_gpu)
                    # try:
                    #     _, loss = model(*example_gpu)
                    # except:
                    #     print(doc_key)
                    #     for item in example_gpu:
                    #         print(item)
                    #     print("==" * 50)

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
                                        (len(loss_history), avg_loss,
                                         conf['report_frequency'] / (end_time - start_time)))
                            start_time = end_time

                            tb_writer.add_scalar('Training_Loss', avg_loss, len(loss_history))
                            tb_writer.add_scalar('Learning_Rate_Bert', schedulers[0].get_last_lr()[0],
                                                 len(loss_history))
                            tb_writer.add_scalar('Learning_Rate_Task', schedulers[1].get_last_lr()[-1],
                                                 len(loss_history))

                        # Evaluate
                        if len(loss_history) > 0 and len(loss_history) % conf['eval_frequency'] == 0:
                            print("Dev Scores:")
                            # f1, _ = self.evaluate(model, examples_dev, stored_info, len(loss_history), official=False, conll_path=self.config['conll_eval_path'], tb_writer=tb_writer)
                            f1, _ = self.evaluate(model, examples_dev, stored_info, len(loss_history), official=False,
                                                  tb_writer=tb_writer)
                            if f1 > max_f1:
                                max_f1 = f1
                                if self.save_checkpoint:
                                    self.save_model_checkpoint(model, len(loss_history))
                            logger.info('Eval max f1: %.2f' % max_f1)
                            start_time = time.time()
                except:
                    print("Error")
        logger.info('**********Finished training**********')
        logger.info('Actual update steps: %d' % len(loss_history))

        # Wrap up
        tb_writer.close()
        return loss_history


    # def train(self, model):
    #     conf = self.config
    #     logger.info(conf)
    #     epochs, grad_accum = conf['num_epochs'], conf['gradient_accumulation_steps']
    #
    #     model.to(self.device)
    #     logger.info('Model parameters:')
    #     for name, param in model.named_parameters():
    #         logger.info('%s: %s' % (name, tuple(param.shape)))
    #
    #     # Set up tensorboard
    #     tb_path = join(conf['tb_dir'], conf['dataset'] + '_' + conf['language'] + '_' + self.name + '_' + self.name_suffix)
    #     tb_writer = SummaryWriter(tb_path, flush_secs=30)
    #     logger.info('Tensorboard summary path: %s' % tb_path)
    #
    #     # Set up data
    #     examples_train, examples_dev, examples_test = self.data.get_tensor_examples()
    #     stored_info = self.data.get_stored_info()
    #
    #     # Set up optimizer and scheduler
    #     total_update_steps = len(examples_train) * epochs // grad_accum
    #     optimizers = self.get_optimizer(model)
    #     schedulers = self.get_scheduler(optimizers, total_update_steps)
    #
    #     # Get model parameters for grad clipping
    #     # Need to modify this to adapt to Chinese and Farsi
    #     bert_param, task_param = model.get_params()
    #
    #     # Start training
    #     logger.info('*******************Training*******************')
    #     logger.info('Num samples: %d' % len(examples_train))
    #     logger.info('Num epochs: %d' % epochs)
    #     logger.info('Gradient accumulation steps: %d' % grad_accum)
    #     logger.info('Total update steps: %d' % total_update_steps)
    #
    #     loss_during_accum = []  # To compute effective loss at each update
    #     loss_during_report = 0.0  # Effective loss during logging step
    #     loss_history = []  # Full history of effective loss; length equals total update steps
    #     max_f1 = 0
    #     start_time = time.time()
    #     model.zero_grad()
    #     for epo in range(epochs):
    #         random.shuffle(examples_train)  # Shuffle training set
    #         for doc_key, example in examples_train:
    #             if example[-3].shape[0]==0:
    #                 continue
    #             # print(example[-3].shape[0])
    #             # Forward pass
    #             model.train()
    #             example_gpu = [d.to(self.device) for d in example]
    #             # try:
    #             #     _, loss = model(*example_gpu)
    #             # except:
    #             #     print("Wrong")
    #             try:
    #                 _, loss = model(*example_gpu)
    #             except:
    #                 print(doc_key)
    #                 for item in example_gpu:
    #                     print(item)
    #                 print("=="*50)
    #
    #             # Backward; accumulate gradients and clip by grad norm
    #             if grad_accum > 1:
    #                 loss /= grad_accum
    #             loss.backward()
    #             if conf['max_grad_norm']:
    #                 torch.nn.utils.clip_grad_norm_(bert_param, conf['max_grad_norm'])
    #                 torch.nn.utils.clip_grad_norm_(task_param, conf['max_grad_norm'])
    #             loss_during_accum.append(loss.item())
    #
    #             # Update
    #             if len(loss_during_accum) % grad_accum == 0:
    #                 for optimizer in optimizers:
    #                     optimizer.step()
    #                 model.zero_grad()
    #                 for scheduler in schedulers:
    #                     scheduler.step()
    #
    #                 # Compute effective loss
    #                 effective_loss = np.sum(loss_during_accum).item()
    #                 loss_during_accum = []
    #                 loss_during_report += effective_loss
    #                 loss_history.append(effective_loss)
    #
    #                 # Report
    #                 if len(loss_history) % conf['report_frequency'] == 0:
    #                     # Show avg loss during last report interval
    #                     avg_loss = loss_during_report / conf['report_frequency']
    #                     loss_during_report = 0.0
    #                     end_time = time.time()
    #                     logger.info('Step %d: avg loss %.2f; steps/sec %.2f' %
    #                                 (len(loss_history), avg_loss, conf['report_frequency'] / (end_time - start_time)))
    #                     start_time = end_time
    #
    #                     tb_writer.add_scalar('Training_Loss', avg_loss, len(loss_history))
    #                     tb_writer.add_scalar('Learning_Rate_Bert', schedulers[0].get_last_lr()[0], len(loss_history))
    #                     tb_writer.add_scalar('Learning_Rate_Task', schedulers[1].get_last_lr()[-1], len(loss_history))
    #
    #                 # Evaluate
    #                 if len(loss_history) > 0 and len(loss_history) % conf['eval_frequency'] == 0:
    #                     print("Dev Scores:")
    #                     # f1, _ = self.evaluate(model, examples_dev, stored_info, len(loss_history), official=False, conll_path=self.config['conll_eval_path'], tb_writer=tb_writer)
    #                     f1, _ = self.evaluate(model, examples_dev, stored_info, len(loss_history), official=False, tb_writer=tb_writer)
    #                     if f1 > max_f1:
    #                         max_f1 = f1
    #                         if self.save_checkpoint:
    #                             self.save_model_checkpoint(model, len(loss_history))
    #                     logger.info('Eval max f1: %.2f' % max_f1)
    #                     start_time = time.time()
    #                 # # Test
    #                 # if len(loss_history) > 0 and len(loss_history) % conf['eval_frequency'] == 0:
    #                 #     print("Test Scores:")
    #                 #     f1, _ = self.evaluate(model, examples_test, stored_info, len(loss_history), official=False, conll_path=self.config['conll_test_path'], tb_writer=tb_writer)
    #                 #     start_time = time.time()
    #                 # # Train
    #                 # if len(loss_history) > 0 and len(loss_history) % conf['eval_frequency'] == 0:
    #                 #     print("Train Scores:")
    #                 #     f1, _ = self.evaluate(model, examples_train, stored_info, len(loss_history), official=False, conll_path=self.config['conll_train_path'], tb_writer=tb_writer)
    #                 #     start_time = time.time()
    #
    #     logger.info('**********Finished training**********')
    #     logger.info('Actual update steps: %d' % len(loss_history))
    #
    #     # Wrap up
    #     tb_writer.close()
    #     return loss_history


    def my_predict(self, model, tensor_examples, stored_info, step, official=False, conll_path=None, tb_writer=None):
        logger.info('Step %d: evaluating on %d samples...' % (step, len(tensor_examples)))
        model.to(self.device)
        evaluator = CorefEvaluator()
        mention_evaluator = MentionScorer()
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
                _, _, _, span_starts, span_ends, antecedent_idx, antecedent_scores, span_scores = model(*example_gpu)
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

    def predict_same_lemma_baseline_golden_boundaries(self, tensor_examples, stored_info, step, tokenizer, wnl=None, official=False, conll_path=None, tb_writer=None):
        evaluator = CorefEvaluator()
        mention_evaluator = MentionScorer()
        doc_to_prediction = {}

        for i, (doc_key, tensor_example) in enumerate(tensor_examples):
            gold_clusters = stored_info['gold'][doc_key]
            input_ids = tensor_example[0][0].cpu().numpy().tolist()

            # Preprocess data
            golden_mentions = []
            for cluster in gold_clusters:
                for mention in cluster:
                    golden_mentions.append(mention)
            golden_mentions.sort()
            # Perform Lemmatization
            lemma_mentions = []
            for start_id, end_id in golden_mentions:
                mention_str = tokenizer.decode(input_ids[start_id: end_id + 1]).lower()
                mention_tokens = mention_str.strip().split(" ")
                if wnl:
                    lemma_tokens = [wnl.lemmatize(token) for token in mention_tokens]
                else:
                    lemma_tokens = mention_tokens
                lemma_mentions.append(lemma_tokens)

            def lemma_coref(mention_a, mention_b):
                max_prefix_num = 0
                for i in range(min(len(mention_a), len(mention_b))):
                    if mention_a[i] == mention_b[i]:
                        max_prefix_num = i + 1
                return max_prefix_num

            def flatten_clusters(clusters):
                temp_mentions = []
                for cluster in clusters:
                    temp_mentions.extend(cluster)
                return temp_mentions

            def rule_based_clustering(lemma_mentions, golden_mentions):
                linkages = []
                for i in range(len(lemma_mentions)):
                    temp_linkage = [i]
                    for j in range(i + 1, len(lemma_mentions)):
                        mention_a = lemma_mentions[i]
                        mention_b = lemma_mentions[j]
                        max_prefix_lemma = lemma_coref(mention_a, mention_b)
                        if max_prefix_lemma != 0:
                            temp_linkage.append(j)
                    linkages.append(temp_linkage)

                # Merge clusters if any clusters have common mentions
                merged_clusters = []
                for cluster in linkages:
                    existing = None
                    for mention in cluster:
                        for merged_cluster in merged_clusters:
                            if mention in merged_cluster:
                                existing = merged_cluster
                                break
                        if existing is not None:
                            break
                    if existing is not None:
                        # print("Merging clusters (shouldn't happen very often)")
                        existing.update(cluster)
                    else:
                        merged_clusters.append(set(cluster))
                # Prepare predicted_clusters
                predicted_clusters = []
                for cluster in merged_clusters:
                    temp = []
                    for mention_id in cluster:
                        temp.append(tuple(golden_mentions[mention_id]))
                    predicted_clusters.append(tuple(temp))
                # print(predicted_clusters)

                # Prepare mention_to_cluster_id
                mention_to_cluster_id = {}
                for i, cluster in enumerate(predicted_clusters):
                    for mention in cluster:
                        mention_to_cluster_id[mention] = i
                # print(mention_to_cluster_id)
                return predicted_clusters, mention_to_cluster_id

            predicted_clusters, mention_to_cluster_id = rule_based_clustering(lemma_mentions, golden_mentions)
            mention_to_predicted = {m: predicted_clusters[cluster_idx] for m, cluster_idx in mention_to_cluster_id.items()}
            gold_clusters = [tuple(tuple(m) for m in cluster) for cluster in gold_clusters]
            mention_to_gold = {m: cluster for cluster in gold_clusters for m in cluster}
            evaluator.update(predicted_clusters, gold_clusters, mention_to_predicted, mention_to_gold)
            mention_evaluator.update(predicted_clusters, gold_clusters, mention_to_predicted, mention_to_gold)
            predictions = {
                "predicted_clusters": predicted_clusters,
                "gold_clusters": gold_clusters,
                "mention_to_predicted": mention_to_predicted,
                "mention_to_gold": mention_to_gold
            }
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

        metric_name = "mention"
        p, r, f = mention_evaluator.get_metric()
        print('%s:  %.2f  %.2f  %.2f' % (metric_name.ljust(10, " "), p * 100, r * 100, f * 100))

        return doc_to_prediction


    def predict_same_lemma_baseline_constituent_boundaries_prefix_match(self, tensor_examples, stored_info, step, tokenizer, mention_candidate_dict, wnl=None, official=False, conll_path=None, tb_writer=None):
        evaluator = CorefEvaluator()
        mention_evaluator = MentionScorer()
        doc_to_prediction = {}

        for i, (doc_key, tensor_example) in enumerate(tensor_examples):
            gold_clusters = stored_info['gold'][doc_key]
            input_ids = tensor_example[0][0].cpu().numpy().tolist()

            # Load candidate mentions and perform preprocessing
            candidate_mentions = mention_candidate_dict[doc_key]
            candidate_mentions.sort()

            # Perform Lemmatization
            lemma_mentions = []
            for start_id, end_id in candidate_mentions:
                mention_str = tokenizer.decode(input_ids[start_id: end_id + 1]).lower()
                mention_tokens = mention_str.strip().split(" ")
                if wnl:
                    lemma_tokens = [wnl.lemmatize(token) for token in mention_tokens]
                else:
                    lemma_tokens = mention_tokens
                lemma_mentions.append(lemma_tokens)

            def lemma_coref(mention_a, mention_b):

                max_prefix_num = 0
                for i in range(min(len(mention_a), len(mention_b))):
                    if mention_a[i] == mention_b[i]:
                        max_prefix_num = i + 1
                return max_prefix_num

            def flatten_clusters(clusters):
                temp_mentions = []
                for cluster in clusters:
                    temp_mentions.extend(cluster)
                return temp_mentions

            def rule_based_clustering(lemma_mentions, golden_mentions):
                linkages = []
                for i in range(len(lemma_mentions)):
                    temp_linkage = [i]
                    for j in range(i + 1, len(lemma_mentions)):
                        mention_a = lemma_mentions[i]
                        mention_b = lemma_mentions[j]
                        max_prefix_lemma = lemma_coref(mention_a, mention_b)
                        if max_prefix_lemma != 0:
                            temp_linkage.append(j)
                    linkages.append(temp_linkage)

                # Merge clusters if any clusters have common mentions
                merged_clusters = []
                for cluster in linkages:
                    existing = None
                    for mention in cluster:
                        for merged_cluster in merged_clusters:
                            if mention in merged_cluster:
                                existing = merged_cluster
                                break
                        if existing is not None:
                            break
                    if existing is not None:
                        # print("Merging clusters (shouldn't happen very often)")
                        existing.update(cluster)
                    else:
                        merged_clusters.append(set(cluster))
                # Prepare predicted_clusters
                predicted_clusters = []
                for cluster in merged_clusters:
                    temp = []
                    for mention_id in cluster:
                        temp.append(tuple(golden_mentions[mention_id]))
                    predicted_clusters.append(tuple(temp))
                # print(predicted_clusters)

                # Prepare mention_to_cluster_id
                mention_to_cluster_id = {}
                for i, cluster in enumerate(predicted_clusters):
                    for mention in cluster:
                        mention_to_cluster_id[mention] = i
                # print(mention_to_cluster_id)
                return predicted_clusters, mention_to_cluster_id


            predicted_clusters, mention_to_cluster_id = rule_based_clustering(lemma_mentions, candidate_mentions)
            mention_to_predicted = {m: predicted_clusters[cluster_idx] for m, cluster_idx in mention_to_cluster_id.items()}
            gold_clusters = [tuple(tuple(m) for m in cluster) for cluster in gold_clusters]
            mention_to_gold = {m: cluster for cluster in gold_clusters for m in cluster}
            evaluator.update(predicted_clusters, gold_clusters, mention_to_predicted, mention_to_gold)
            mention_evaluator.update(predicted_clusters, gold_clusters, mention_to_predicted, mention_to_gold)
            predictions = {
                "predicted_clusters": predicted_clusters,
                "gold_clusters": gold_clusters,
                "mention_to_predicted": mention_to_predicted,
                "mention_to_gold": mention_to_gold
            }
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

        metric_name = "mention"
        p, r, f = mention_evaluator.get_metric()
        print('%s:  %.2f  %.2f  %.2f' % (metric_name.ljust(10, " "), p * 100, r * 100, f * 100))

        return doc_to_prediction

    def predict_same_lemma_baseline_constituent_boundaries(self, tensor_examples, stored_info, step, tokenizer,
                                                           mention_candidate_dict, wnl=None, official=False,
                                                           conll_path=None, tb_writer=None):
        evaluator = CorefEvaluator()
        mention_evaluator = MentionScorer()
        doc_to_prediction = {}

        for i, (doc_key, tensor_example) in enumerate(tensor_examples):
            gold_clusters = stored_info['gold'][doc_key]
            input_ids = tensor_example[0][0].cpu().numpy().tolist()

            # Load candidate mentions and perform preprocessing
            candidate_mentions = mention_candidate_dict[doc_key]
            candidate_mentions.sort()

            # Perform Lemmatization
            lemma_mentions = []
            for start_id, end_id in candidate_mentions:
                mention_str = tokenizer.decode(input_ids[start_id: end_id + 1]).lower()
                mention_tokens = mention_str.strip().split(" ")

                # Extract Head-Words and Perform Lemmatization
                doc = wnl(mention_str)
                lemma_head = mention_tokens[0]
                for token in doc:
                    if token.text==token.head.text:
                        lemma_head = token.lemma_
                        break
                lemma_mentions.append(lemma_head)

            def lemma_coref(mention_a, mention_b):
                return mention_a==mention_b

            def flatten_clusters(clusters):
                temp_mentions = []
                for cluster in clusters:
                    temp_mentions.extend(cluster)
                return temp_mentions

            def rule_based_clustering(lemma_mentions, golden_mentions):
                linkages = []
                for i in range(len(lemma_mentions)):
                    temp_linkage = [i]
                    for j in range(i + 1, len(lemma_mentions)):
                        mention_a = lemma_mentions[i]
                        mention_b = lemma_mentions[j]
                        if lemma_coref(mention_a, mention_b):
                            temp_linkage.append(j)
                    linkages.append(temp_linkage)

                # Merge clusters if any clusters have common mentions
                merged_clusters = []
                for cluster in linkages:
                    existing = None
                    for mention in cluster:
                        for merged_cluster in merged_clusters:
                            if mention in merged_cluster:
                                existing = merged_cluster
                                break
                        if existing is not None:
                            break
                    if existing is not None:
                        # print("Merging clusters (shouldn't happen very often)")
                        existing.update(cluster)
                    else:
                        merged_clusters.append(set(cluster))
                # Prepare predicted_clusters
                predicted_clusters = []
                for cluster in merged_clusters:
                    temp = []
                    for mention_id in cluster:
                        temp.append(tuple(golden_mentions[mention_id]))
                    predicted_clusters.append(tuple(temp))
                # print(predicted_clusters)

                # Prepare mention_to_cluster_id
                mention_to_cluster_id = {}
                for i, cluster in enumerate(predicted_clusters):
                    for mention in cluster:
                        mention_to_cluster_id[mention] = i
                # print(mention_to_cluster_id)
                return predicted_clusters, mention_to_cluster_id

            predicted_clusters, mention_to_cluster_id = rule_based_clustering(lemma_mentions, candidate_mentions)
            mention_to_predicted = {m: predicted_clusters[cluster_idx] for m, cluster_idx in
                                    mention_to_cluster_id.items()}
            gold_clusters = [tuple(tuple(m) for m in cluster) for cluster in gold_clusters]
            mention_to_gold = {m: cluster for cluster in gold_clusters for m in cluster}
            evaluator.update(predicted_clusters, gold_clusters, mention_to_predicted, mention_to_gold)
            mention_evaluator.update(predicted_clusters, gold_clusters, mention_to_predicted, mention_to_gold)
            predictions = {
                "predicted_clusters": predicted_clusters,
                "gold_clusters": gold_clusters,
                "mention_to_predicted": mention_to_predicted,
                "mention_to_gold": mention_to_gold
            }
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

        metric_name = "mention"
        p, r, f = mention_evaluator.get_metric()
        print('%s:  %.2f  %.2f  %.2f' % (metric_name.ljust(10, " "), p * 100, r * 100, f * 100))

        return doc_to_prediction

    def predict(self, model, tensor_examples, stored_info, step, official=False, conll_path=None, tb_writer=None):
        model.to(self.device)
        evaluator = CorefEvaluator()
        mention_evaluator = MentionScorer()
        doc_to_prediction = {}

        model.eval()
        for i, (doc_key, tensor_example) in tqdm(enumerate(tensor_examples)):
            gold_clusters = stored_info['gold'][doc_key]
            tensor_example = tensor_example  # Strip out gold
            # tensor_example = tensor_example[:7]  # Strip out gold
            example_gpu = [d.to(self.device) for d in tensor_example]
            try:
                with torch.no_grad():
                    _, _, _, span_starts, span_ends, antecedent_idx, antecedent_scores, span_scores = model(*example_gpu, False)
                span_starts, span_ends = span_starts.tolist(), span_ends.tolist()
                antecedent_idx, antecedent_scores = antecedent_idx.tolist(), antecedent_scores.tolist()
                # predicted_clusters = model.update_evaluator(span_starts, span_ends, antecedent_idx, antecedent_scores, gold_clusters, evaluator)
                predicted_clusters, predictions = model.update_evaluator_prediction(span_starts, span_ends,
                                                                                    antecedent_idx, antecedent_scores,
                                                                                    gold_clusters, span_scores,
                                                                                    evaluator, mention_evaluator)
                doc_to_prediction[doc_key] = predictions
            except:
                print(doc_key)

            # with torch.no_grad():
            #     _, _, _, span_starts, span_ends, antecedent_idx, antecedent_scores, span_scores = model(*example_gpu,
            #                                                                                             False)
            # span_starts, span_ends = span_starts.tolist(), span_ends.tolist()
            # antecedent_idx, antecedent_scores = antecedent_idx.tolist(), antecedent_scores.tolist()
            # # predicted_clusters = model.update_evaluator(span_starts, span_ends, antecedent_idx, antecedent_scores, gold_clusters, evaluator)
            # predicted_clusters, predictions = model.update_evaluator_prediction(span_starts, span_ends,
            #                                                                     antecedent_idx, antecedent_scores,
            #                                                                     gold_clusters, span_scores,
            #                                                                     evaluator, mention_evaluator)
            # doc_to_prediction[doc_key] = predictions

            # try:
            #     _, _, _, span_starts, span_ends, antecedent_idx, antecedent_scores, span_scores = model(*example_gpu, False)
            #     span_starts, span_ends = span_starts.tolist(), span_ends.tolist()
            #     antecedent_idx, antecedent_scores = antecedent_idx.tolist(), antecedent_scores.tolist()
            #     # predicted_clusters = model.update_evaluator(span_starts, span_ends, antecedent_idx, antecedent_scores, gold_clusters, evaluator)
            #     predicted_clusters, predictions = model.update_evaluator_prediction(span_starts, span_ends,
            #                                                                         antecedent_idx, antecedent_scores,
            #                                                                         gold_clusters, span_scores,
            #                                                                         evaluator, mention_evaluator)
            #     doc_to_prediction[doc_key] = predictions
            # except:
            #     print(doc_key)
            # # with torch.no_grad():
            # #     # try:
            # #     #     _, _, _, span_starts, span_ends, antecedent_idx, antecedent_scores, span_scores = model(
            # #     #         *example_gpu)
            # #     # except:
            # #     #     print("Error", doc_key)
            # #     _, _, _, span_starts, span_ends, antecedent_idx, antecedent_scores, span_scores = model(*example_gpu, False)
            # span_starts, span_ends = span_starts.tolist(), span_ends.tolist()
            # antecedent_idx, antecedent_scores = antecedent_idx.tolist(), antecedent_scores.tolist()
            # # predicted_clusters = model.update_evaluator(span_starts, span_ends, antecedent_idx, antecedent_scores, gold_clusters, evaluator)
            # predicted_clusters, predictions = model.update_evaluator_prediction(span_starts, span_ends, antecedent_idx, antecedent_scores, gold_clusters, span_scores, evaluator, mention_evaluator)
            # doc_to_prediction[doc_key] = predictions

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

        metric_name = "mention"
        p, r, f = mention_evaluator.get_metric()
        print('%s:  %.2f  %.2f  %.2f' % (metric_name.ljust(10, " "), p * 100, r * 100, f * 100))
        # p, r, f = evaluator.get_prf()
        # metrics = {'Eval_Avg_Precision': p * 100, 'Eval_Avg_Recall': r * 100, 'Eval_Avg_F1': f * 100}
        # for name, score in metrics.items():
        #     logger.info('%s: %.2f' % (name, score))
        #     if tb_writer:
        #         tb_writer.add_scalar(name, score, step)

        # if official:
        #     conll_results = conll.evaluate_conll(conll_path, doc_to_prediction, stored_info['subtoken_maps'])
        #     official_f1 = sum(results["f"] for results in conll_results.values()) / len(conll_results)
        #     logger.info('Official avg F1: %.4f' % official_f1)

        return doc_to_prediction


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
        # path_ckpt = join(self.config['log_dir'], f'model_{self.name_suffix}_{step}.bin')
        path_ckpt = join(self.config['log_dir'], f'model_{self.name_suffix}_best.bin')
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

    config_name, gpu_id, random_seed = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
    if gpu_id == 10:
        gpu_id = -1

    # Set Random Seeds
    def setup_seed(seed):
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        random.seed(seed)
        torch.backends.cudnn.deterministic = True
    setup_seed(random_seed)

    print("Random Seeds:", random_seed)

    runner = Runner(config_name, gpu_id)
    model = runner.initialize_model()

    runner.train(model)

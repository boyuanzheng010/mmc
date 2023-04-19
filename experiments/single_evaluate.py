from run import Runner
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
from metrics import CorefEvaluator
from datetime import datetime
from torch.optim.lr_scheduler import LambdaLR
from model import CorefModel
import conll
import sys

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger()

config_name = "train_spanbert_large_ml0_d1"
gpu_id = 5
saved_suffix = "May08_12-37-39_54000"

runner = Runner(config_name, gpu_id)
model = runner.initialize_model(saved_suffix)

examples_train, examples_dev, examples_test = runner.data.get_tensor_examples()
stored_info = runner.data.get_stored_info()

# runner.evaluate(model, examples_test, stored_info, 0, official=False, conll_path=runner.config['conll_test_path'])
evaluator = CorefEvaluator()
doc_to_prediction = {}
model.eval()

for i, (doc_key, tensor_example) in enumerate(examples_test):
    gold_clusters = stored_info['gold'][doc_key]
    tensor_example = tensor_example[:7]  # Strip out gold
    # example_gpu = [d.to(self.device) for d in tensor_example]
    with torch.no_grad():
        _, _, _, span_starts, span_ends, antecedent_idx, antecedent_scores = model(*tensor_example)
    print("Index:", i)
    print("span_start:", span_starts)
    print("span_ends:", span_ends)
    print("antecedent_idx:", antecedent_idx)
    print("antecedent_scores:", antecedent_scores)
    print("==" * 50)
    span_starts, span_ends = span_starts.tolist(), span_ends.tolist()
    antecedent_idx, antecedent_scores = antecedent_idx.tolist(), antecedent_scores.tolist()
    predicted_clusters = model.update_evaluator(span_starts, span_ends, antecedent_idx, antecedent_scores,
                                                gold_clusters, evaluator)
    doc_to_prediction[doc_key] = predicted_clusters


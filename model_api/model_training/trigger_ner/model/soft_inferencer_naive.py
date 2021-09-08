"""soft_inferencer_navie.py: Inference on Unlabeled Sentences (without trigger - baseline setting)
Baseline setting inference. (normal BLSTM-CRF.)

Written in 2020 by Dong-Ho Lee.
"""
import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
import logging
import time
import sys
import pathlib

from ..utilities.config import ContextEmb, Config
from ..utilities.utils import get_optimizer, batching_list_instances
from ..utilities.eval import evaluate_batch_insts
from .linear_crf_inferencer import LinearCRF
from .soft_encoder import SoftEncoder

PATH_TO_PARENT = str(pathlib.Path(__file__).parent.absolute()) + "/"
# sys.path.append(PATH_TO_PARENT)
# sys.path.append(PATH_TO_PARENT + "../")
sys.path.append(PATH_TO_PARENT + "../../../")
from fast_api.fast_api_util_functions import update_model_training


class SoftSequenceNaive(nn.Module):
    def __init__(self, config: Config, encoder=None, print_info=True):
        super(SoftSequenceNaive, self).__init__()
        self.config = config
        self.device = config.device

        self.encoder = SoftEncoder(self.config)
        if encoder is not None:
            self.encoder = encoder

        self.label_size = config.label_size
        self.inferencer = LinearCRF(config, print_info=print_info)
        self.hidden2tag = nn.Linear(config.hidden_dim, self.label_size).to(self.device)

    def forward(self, word_seq_tensor: torch.Tensor,
                word_seq_lens: torch.Tensor,
                batch_context_emb: torch.Tensor,
                char_inputs: torch.Tensor,
                char_seq_lens: torch.Tensor, tags):

        batch_size = word_seq_tensor.size(0)
        max_sent_len = word_seq_tensor.size(1)

        output, sentence_mask, _, _ = \
            self.encoder(word_seq_tensor, word_seq_lens, batch_context_emb, char_inputs, char_seq_lens, None)

        lstm_scores = self.hidden2tag(output)
        maskTemp = torch.arange(1, max_sent_len + 1, dtype=torch.long).view(1, max_sent_len).expand(batch_size,
                                                                                                    max_sent_len).to(
            self.device)
        mask = torch.le(maskTemp, word_seq_lens.view(batch_size, 1).expand(batch_size, max_sent_len)).to(self.device)

        if self.inferencer is not None:
            unlabeled_score, labeled_score = self.inferencer(lstm_scores, word_seq_lens, tags, mask)
            sequence_loss = unlabeled_score - labeled_score
        else:
            sequence_loss = self.compute_nll_loss(lstm_scores, tags, mask, word_seq_lens)

        return sequence_loss

    def decode(self, word_seq_tensor: torch.Tensor,
               word_seq_lens: torch.Tensor,
               batch_context_emb: torch.Tensor,
               char_inputs: torch.Tensor,
               char_seq_lens: torch.Tensor):

        soft_output, soft_sentence_mask, _, _ = \
            self.encoder(word_seq_tensor, word_seq_lens, batch_context_emb, char_inputs, char_seq_lens, None)
        lstm_scores = self.hidden2tag(soft_output)
        if self.inferencer is not None:
            bestScores, decodeIdx = self.inferencer.decode(lstm_scores, word_seq_lens, None)
        return bestScores, decodeIdx


class SoftSequenceNaiveTrainer(object):
    def __init__(self, model, config: Config, dev=None, test=None):
        self.model = model
        self.config = config
        self.device = config.device
        self.input_size = config.embedding_dim
        self.context_emb = config.context_emb
        self.use_char = config.use_char_rnn
        if self.context_emb != ContextEmb.none:
            self.input_size += config.context_emb_size
        if self.use_char:
            self.input_size += config.charlstm_hidden_dim
        self.dev = dev
        self.test = test

    def save_model(self):
        logging.info("Saving Model")
        model_save_path = self.config.generate_model_path("naive")
        torch.save(self.model.state_dict(), model_save_path)

    def train_model(self, num_epochs, train_data):
        batched_data = batching_list_instances(self.config, train_data)
        self.optimizer = get_optimizer(self.config, self.model, self.config.optimizer)
        start_time = time.time()
        best_train_loss = 1e30
        best_dev_f1_score = -1

        for epoch in range(num_epochs):
            epoch_loss = 0
            self.model.zero_grad()
            for index in tqdm(np.random.permutation(len(batched_data))):
                self.model.train()
                sequence_loss = self.model(*batched_data[index][0:5], batched_data[index][-3])
                loss = sequence_loss
                epoch_loss = epoch_loss + loss.data
                loss.backward(retain_graph=True)
                self.optimizer.step()
                self.model.zero_grad()
            logging.info(epoch_loss)
            best_train_loss = min(best_train_loss, epoch_loss.item())

            self.model.eval()

            if self.dev:
                dev_batches = batching_list_instances(self.config, self.dev)
                dev_metrics = self.evaluate_model(dev_batches, "dev", self.dev)
                if best_dev_f1_score < dev_metrics[2]:
                    self.save_model()
                    best_dev_f1_score = dev_metrics[2]
            else:
                self.save_model()

            if self.test:
                test_batches = batching_list_instances(self.config, self.test)
                test_metrics = self.evaluate_model(test_batches, "test", self.test)
            self.model.zero_grad()

            if self.config.is_lean_life:
                time_spent = time.time() - start_time
                update_model_training(
                    self.config.project_id, self.config.experiment_name,
                    epoch + 1, num_epochs,
                    time_spent, epoch_loss,
                    "training"
                )
        return self.model, best_train_loss

    def evaluate_model(self, batch_insts_ids, name: str, insts):
        ## evaluation
        metrics = np.asarray([0, 0, 0], dtype=int)
        batch_id = 0
        batch_size = self.config.batch_size
        for batch in batch_insts_ids:
            one_batch_insts = insts[batch_id * batch_size:(batch_id + 1) * batch_size]
            batch_max_scores, batch_max_ids = self.model.decode(*batch[0:5])
            metrics += evaluate_batch_insts(one_batch_insts, batch_max_ids, batch[6], batch[1], self.config.idx2labels,
                                            self.config.use_crf_layer)
            batch_id += 1
        p, total_predict, total_entity = metrics[0], metrics[1], metrics[2]
        precision = p * 1.0 / total_predict * 100 if total_predict != 0 else 0
        recall = p * 1.0 / total_entity * 100 if total_entity != 0 else 0
        fscore = 2.0 * precision * recall / (precision + recall) if precision != 0 or recall != 0 else 0
        logging.info("[%s set] Precision: %.2f, Recall: %.2f, F1: %.2f" % (name, precision, recall, fscore))
        return [precision, recall, fscore]

    def predict_model(self, batch_insts_ids, insts):
        ## prediction
        batch_id = 0
        batch_size = self.config.batch_size
        for batch in batch_insts_ids:
            one_batch_insts = insts[batch_id * batch_size:(batch_id + 1) * batch_size]
            batch_max_scores, batch_max_ids = self.model.decode(*batch[0:5])
            word_seq_lens = batch[1].tolist()
            for idx in range(len(batch_max_ids)):
                length = word_seq_lens[idx]
                prediction = batch_max_ids[idx][:length].tolist()
                prediction = prediction[::-1] if self.config.use_crf_layer else prediction
                prediction = [self.config.idx2labels[l] for l in prediction]
                one_batch_insts[idx].prediction = prediction
            batch_id += 1

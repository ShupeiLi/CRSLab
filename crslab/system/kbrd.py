# -*- encoding: utf-8 -*-
# @Time    :   2020/12/4
# @Author  :   Xiaolei Wang
# @email   :   wxl1999@foxmail.com

# UPDATE
# @Time    :   2021/1/3
# @Author  :   Xiaolei Wang
# @email   :   wxl1999@foxmail.com

import os

import torch
from loguru import logger

from crslab.evaluator.metrics.base import AverageMetric
from crslab.evaluator.metrics.gen import PPLMetric
from crslab.system.base import BaseSystem
from crslab.system.utils.functions import ind2txt


class KBRDSystem(BaseSystem):
    """This is the system for KBRD model"""

    def __init__(self, opt, train_dataloader, valid_dataloader, test_dataloader, vocab, side_data, restore_system=False,
                 interact=False, debug=False, tensorboard=False, test_only=False):
        """

        Args:
            opt (dict): Indicating the hyper parameters.
            train_dataloader (BaseDataLoader): Indicating the train dataloader of corresponding dataset.
            valid_dataloader (BaseDataLoader): Indicating the valid dataloader of corresponding dataset.
            test_dataloader (BaseDataLoader): Indicating the test dataloader of corresponding dataset.
            vocab (dict): Indicating the vocabulary.
            side_data (dict): Indicating the side data.
            restore_system (bool, optional): Indicating if we store system after training. Defaults to False.
            interact (bool, optional): Indicating if we interact with system. Defaults to False.
            debug (bool, optional): Indicating if we train in debug mode. Defaults to False.
            tensorboard (bool, optional) Indicating if we monitor the training performance in tensorboard. Defaults to False. 

        """
        super(KBRDSystem, self).__init__(opt, train_dataloader, valid_dataloader, test_dataloader, vocab, side_data,
                                         restore_system, interact, debug, tensorboard, test_only)

        self.ind2tok = vocab['ind2tok']
        self.unk = vocab['unk']
        self.end_token_idx = vocab['end']
        self.item_ids = side_data['item_entity_ids']

        self.rec_optim_opt = opt['rec']
        self.conv_optim_opt = opt['conv']
        self.rec_epoch = self.rec_optim_opt['epoch']
        self.conv_epoch = self.conv_optim_opt['epoch']
        self.rec_batch_size = self.rec_optim_opt['batch_size']
        self.conv_batch_size = self.conv_optim_opt['batch_size']

    def rec_evaluate(self, rec_predict, item_label):
        rec_predict = rec_predict.cpu()
        rec_predict = rec_predict[:, self.item_ids]
        _, rec_ranks = torch.topk(rec_predict, 50, dim=-1)
        rec_ranks = rec_ranks.tolist()
        item_label = item_label.tolist()
        for rec_rank, label in zip(rec_ranks, item_label):
            label = self.item_ids.index(label)
            self.evaluator.rec_evaluate(rec_rank, label)

    def conv_evaluate(self, prediction, response):
        prediction = prediction.tolist()
        response = response.tolist()
        for p, r in zip(prediction, response):
            p_str = ind2txt(p, self.ind2tok, self.end_token_idx)
            r_str = ind2txt(r, self.ind2tok, self.end_token_idx)
            self.evaluator.gen_evaluate(p_str, [r_str])

    def step(self, batch, stage, mode):
        assert stage in ('rec', 'conv')
        assert mode in ('train', 'valid', 'test')

        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                batch[k] = v.to(self.device)

        if stage == 'rec':
            rec_loss, rec_scores = self.model.forward(batch, mode, stage)
            rec_loss = rec_loss.sum()
            if mode == 'train':
                self.backward(rec_loss)
            else:
                self.rec_evaluate(rec_scores, batch['item'])
            rec_loss = rec_loss.item()
            self.evaluator.optim_metrics.add("rec_loss", AverageMetric(rec_loss))
        else:
            if mode != 'test':
                gen_loss, preds = self.model.forward(batch, mode, stage)
                if mode == 'train':
                    self.backward(gen_loss)
                else:
                    self.conv_evaluate(preds, batch['response'])
                gen_loss = gen_loss.item()
                self.evaluator.optim_metrics.add('gen_loss', AverageMetric(gen_loss))
                self.evaluator.gen_metrics.add("ppl", PPLMetric(gen_loss))
            else:
                preds = self.model.forward(batch, mode, stage)
                self.conv_evaluate(preds, batch['response'])
                response = batch['response']
                self.record_conv_gt_pred(response, preds)
                self.record_conv_gt(response, preds)
                self.record_conv_pred(response, preds)

    def record_conv_gt_pred(self, batch_response, batch_pred):
        # (bs, response_truncate), (bs, response_truncate)
        file_writer = self.get_file_writer(f'record_conv_gt_pred', '.txt')

        for response, pred in zip(batch_response, batch_pred):
            response_tok_list = self.convert_tensor_ids_to_tokens(response)
            pred_tok_list = self.convert_tensor_ids_to_tokens(pred)

            file_writer.writelines(' '.join(response_tok_list) + '\n')
            file_writer.writelines(' '.join(pred_tok_list) + '\n')
            file_writer.writelines('\n')

        file_writer.close()

    def record_conv_gt(self, batch_response, batch_pred):
        # (bs, response_truncate), (bs, response_truncate)
        file_writer = self.get_file_writer('record_conv_gt', '.txt')

        for response, pred in zip(batch_response, batch_pred):
            response_tok_list = self.convert_tensor_ids_to_tokens(response)

            file_writer.writelines(' '.join(response_tok_list) + '\n')
            file_writer.writelines('\n')

        file_writer.close()

    def record_conv_pred(self, batch_response, batch_pred):
        # (bs, response_truncate), (bs, response_truncate)
        file_writer = self.get_file_writer(f'record_conv_pred', '.txt')

        for response, pred in zip(batch_response, batch_pred):
            pred_tok_list = self.convert_tensor_ids_to_tokens(pred)

            file_writer.writelines(' '.join(pred_tok_list) + '\n')
            file_writer.writelines('\n')

        file_writer.close()

    def convert_tensor_ids_to_tokens(self, token_ids):
        tokens = []

        token_ids = token_ids.tolist() # List[int]
        if not token_ids:
            return tokens

        for token_id in token_ids:
            if token_id == self.end_token_idx:
                return tokens
            tokens.append(self.ind2tok.get(token_id, self.unk))

        return tokens

    def get_file_writer(self, file_keywords: str, file_type: str):
        file_name = file_keywords + file_type
        file_path = os.path.join(self.opt['LOG_PATH'], file_name)
        if os.path.exists(file_path):
            file_writer = open(file_path, 'a', encoding='utf-8')
        else:
            file_writer = open(file_path, 'w', encoding='utf-8')

        return file_writer

    def train_recommender(self):
        self.init_optim(self.rec_optim_opt, self.model.parameters())

        if not self.test_only:
            for epoch in range(self.rec_epoch):
                self.evaluator.reset_metrics()
                logger.info(f'[Recommendation epoch {str(epoch)}]')
                logger.info('[Train]')
                for batch in self.train_dataloader.get_rec_data(self.rec_batch_size):
                    self.step(batch, stage='rec', mode='train')
                self.evaluator.report(epoch=epoch, mode='train')
                # val
                logger.info('[Valid]')
                with torch.no_grad():
                    self.evaluator.reset_metrics()
                    for batch in self.valid_dataloader.get_rec_data(self.rec_batch_size, shuffle=False):
                        self.step(batch, stage='rec', mode='valid')
                    self.evaluator.report(epoch=epoch, mode='valid')
                    # early stop
                    metric = self.evaluator.optim_metrics['rec_loss']
                    save = (epoch == (self.rec_epoch - 1))
                    if self.early_stop(metric, 0, epoch, save):
                        break

        # test
        def test():
            with torch.no_grad():
                self.evaluator.reset_metrics()
                for batch in self.test_dataloader.get_rec_data(self.rec_batch_size, shuffle=False):
                    self.step(batch, stage='rec', mode='test')
                self.evaluator.report(mode='test')

        logger.info('[Test]')
        logger.info('[Test the best model]')
        checkpoint = self._load_checkpoints(0, 'best')
        if checkpoint is not None:
            self.model.load_state_dict(checkpoint)
            test()
        logger.info('[Test the last model]')
        checkpoint = self._load_checkpoints(0, 'last')
        self.model.load_state_dict(checkpoint)
        test()

    def train_conversation(self):
        self.model.freeze_parameters()
        self.init_optim(self.conv_optim_opt, self.model.parameters())

        if not self.test_only:
            for epoch in range(self.conv_epoch):
                self.evaluator.reset_metrics()
                logger.info(f'[Conversation epoch {str(epoch)}]')
                logger.info('[Train]')
                for batch in self.train_dataloader.get_conv_data(batch_size=self.conv_batch_size):
                    self.step(batch, stage='conv', mode='train')
                self.evaluator.report(epoch=epoch, mode='train')
                # val
                logger.info('[Valid]')
                with torch.no_grad():
                    self.evaluator.reset_metrics()
                    for batch in self.valid_dataloader.get_conv_data(batch_size=self.conv_batch_size, shuffle=False):
                        self.step(batch, stage='conv', mode='valid')
                    self.evaluator.report(epoch=epoch, mode='valid')
                    # early stop
                    metric = self.evaluator.optim_metrics['gen_loss']
                    save = (epoch == (self.conv_epoch - 1))
                    if self.early_stop(metric, 1, epoch, save):
                        break

        # test
        def test():
            with torch.no_grad():
                self.evaluator.reset_metrics()
                for batch in self.test_dataloader.get_conv_data(batch_size=self.conv_batch_size, shuffle=False):
                    self.step(batch, stage='conv', mode='test')
                self.evaluator.report(mode='test')

        logger.info('[Test]')
        logger.info('[Test the best model]')
        checkpoint = self._load_checkpoints(1, 'best')
        if checkpoint is not None:
            self.model.load_state_dict(checkpoint)
            test()
        logger.info('[Test the last model]')
        checkpoint = self._load_checkpoints(1, 'last')
        self.model.load_state_dict(checkpoint)
        test()

    def fit(self):
        self.train_recommender()
        checkpoint = self._load_checkpoints(0, 'best')
        if checkpoint is not None:
            self.model.load_state_dict(checkpoint)
        self.train_conversation()

    def interact(self):
        pass

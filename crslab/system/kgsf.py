# @Time   : 2020/11/22
# @Author : Kun Zhou
# @Email  : francis_kun_zhou@163.com

# UPDATE:
# @Time   : 2020/11/24, 2021/1/3
# @Author : Kun Zhou, Xiaolei Wang
# @Email  : francis_kun_zhou@163.com, wxl1999@foxmail.com

import os

import torch
from loguru import logger

from crslab.evaluator.metrics.base import AverageMetric
from crslab.evaluator.metrics.gen import PPLMetric
from crslab.system.base import BaseSystem
from crslab.system.utils.functions import ind2txt


class KGSFSystem(BaseSystem):
    """This is the system for KGSF model"""

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
        super(KGSFSystem, self).__init__(opt, train_dataloader, valid_dataloader, test_dataloader, vocab, side_data,
                                         restore_system, interact, debug, tensorboard, test_only)

        self.ind2tok = vocab['ind2tok']
        self.unk = vocab['unk']
        self.end_token_idx = vocab['end']
        self.item_ids = side_data['item_entity_ids']

        self.pretrain_optim_opt = self.opt['pretrain']
        self.rec_optim_opt = self.opt['rec']
        self.conv_optim_opt = self.opt['conv']
        self.pretrain_epoch = self.pretrain_optim_opt['epoch']
        self.rec_epoch = self.rec_optim_opt['epoch']
        self.conv_epoch = self.conv_optim_opt['epoch']
        self.pretrain_batch_size = self.pretrain_optim_opt['batch_size']
        self.rec_batch_size = self.rec_optim_opt['batch_size']
        self.conv_batch_size = self.conv_optim_opt['batch_size']

    def rec_evaluate(self, rec_predict, item_label):
        rec_predict = rec_predict.cpu()
        rec_predict = rec_predict[:, self.item_ids]
        _, rec_ranks = torch.topk(rec_predict, 50, dim=-1)
        rec_ranks = rec_ranks.tolist()
        item_label = item_label.tolist()
        for rec_rank, item in zip(rec_ranks, item_label):
            item = self.item_ids.index(item)
            self.evaluator.rec_evaluate(rec_rank, item)

    def conv_evaluate(self, prediction, response):
        prediction = prediction.tolist()
        response = response.tolist()
        for p, r in zip(prediction, response):
            p_str = ind2txt(p, self.ind2tok, self.end_token_idx)
            r_str = ind2txt(r, self.ind2tok, self.end_token_idx)
            self.evaluator.gen_evaluate(p_str, [r_str])

    def step(self, batch, stage, mode):
        batch = [ele.to(self.device) for ele in batch]
        if stage == 'pretrain':
            info_loss = self.model.forward(batch, stage, mode)
            if info_loss is not None:
                self.backward(info_loss.sum())
                info_loss = info_loss.sum().item()
                self.evaluator.optim_metrics.add("info_loss", AverageMetric(info_loss))
        elif stage == 'rec':
            rec_loss, info_loss, rec_predict = self.model.forward(batch, stage, mode)
            if info_loss:
                loss = rec_loss + 0.025 * info_loss
            else:
                loss = rec_loss
            if mode == "train":
                self.backward(loss.sum())
            else:
                self.rec_evaluate(rec_predict, batch[-1])
            rec_loss = rec_loss.sum().item()
            self.evaluator.optim_metrics.add("rec_loss", AverageMetric(rec_loss))
            if info_loss:
                info_loss = info_loss.sum().item()
                self.evaluator.optim_metrics.add("info_loss", AverageMetric(info_loss))
        elif stage == "conv":
            if mode != "test":
                gen_loss, pred = self.model.forward(batch, stage, mode)
                if mode == 'train':
                    self.backward(gen_loss.sum())
                else:
                    self.conv_evaluate(pred, batch[-1])
                gen_loss = gen_loss.sum().item()
                self.evaluator.optim_metrics.add("gen_loss", AverageMetric(gen_loss))
                self.evaluator.gen_metrics.add("ppl", PPLMetric(gen_loss))
            else:
                pred = self.model.forward(batch, stage, mode)
                self.conv_evaluate(pred, batch[-1])
                response = batch[-1]
                self.record_conv_gt_pred(response, pred)
                self.record_conv_gt(response, pred)
                self.record_conv_pred(response, pred)
        else:
            raise


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

    def pretrain(self):
        self.init_optim(self.pretrain_optim_opt, self.model.parameters())

        for epoch in range(self.pretrain_epoch):
            self.evaluator.reset_metrics()
            logger.info(f'[Pretrain epoch {str(epoch)}]')
            for batch in self.train_dataloader.get_pretrain_data(self.pretrain_batch_size, shuffle=False):
                self.step(batch, stage="pretrain", mode='train')
            self.evaluator.report()

    def train_recommender(self):
        self.init_optim(self.rec_optim_opt, self.model.parameters())

        if not self.test_only:
            for epoch in range(self.rec_epoch):
                self.evaluator.reset_metrics()
                logger.info(f'[Recommendation epoch {str(epoch)}]')
                logger.info('[Train]')
                for batch in self.train_dataloader.get_rec_data(self.rec_batch_size, shuffle=False):
                    self.step(batch, stage='rec', mode='train')
                self.evaluator.report(epoch=epoch, mode='train')
                # val
                logger.info('[Valid]')
                with torch.no_grad():
                    self.evaluator.reset_metrics()
                    for batch in self.valid_dataloader.get_rec_data(self.rec_batch_size, shuffle=False):
                        self.step(batch, stage='rec', mode='val')
                    self.evaluator.report(epoch=epoch, mode='val')
                    # early stop
                    metric = self.evaluator.rec_metrics['recall@1'] + self.evaluator.rec_metrics['recall@50']
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
                for batch in self.train_dataloader.get_conv_data(batch_size=self.conv_batch_size, shuffle=False):
                    self.step(batch, stage='conv', mode='train')
                self.evaluator.report(epoch=epoch, mode='train')
                # val
                logger.info('[Valid]')
                with torch.no_grad():
                    self.evaluator.reset_metrics()
                    for batch in self.valid_dataloader.get_conv_data(batch_size=self.conv_batch_size, shuffle=False):
                        self.step(batch, stage='conv', mode='val')
                    self.evaluator.report(epoch=epoch, mode='val')
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
        if not self.test_only:
            self.pretrain()
        self.train_recommender()
        self.train_conversation()

    def interact(self):
        pass

import torch
import torch.nn as nn
import torch.nn.functional as F

import src.data.config as cfg
import src.data.data as data
import src.train.utils as train_utils
import src.train.batch as batch

import src.evaluate.evaluate as evaluate
import src.evaluate.generate as gen
import src.evaluate.sampler as sampling

import utils.utils as utils

from tensorboardX import SummaryWriter


class Trainer(object):
    def __init__(self, opt, meta, data_loader, model, model_knowledge, optimizer_m, optimizer_k):
        self.optimizer_m = optimizer_m
        self.optimizer_k = optimizer_k

        self.model = model
        self.model_knowledge = model_knowledge
        
        if opt.trainer == "epoch":
            self.epochs = meta.epochs
        self.data_loader = data_loader
        self.opt = opt

        self.losses_k = {"dev": {}, "test": {}, "train": {}}
        self.losses_s = {"dev": {}, "test": {}, "train": {}}
        self.top_score = None

        self.lrs_k = {}
        self.lrs_m = {}
        self.batch_variables = {
            "data": self.data_loader,
            "model": self.model,
            "model_knowledge": self.model_knowledge,
            "split": "train"
        }

        self.do_gen = cfg.do_gen
        self.samplers = {}

    def decide_to_save(self):
        to_save = cfg.save and not cfg.toy
        to_save = to_save or cfg.test_save
        if cfg.save_strategy == "best":
            if self.top_score[0] != self.opt.train.dynamic.epoch:
                to_save = False
        return to_save

    def save_model(self, tracked_score):
        lrs = {}
        for i, param_group in enumerate(self.optimizer_m.param_groups):
            lrs[i] = param_group['lr']
        self.lrs_m[self.opt.train.dynamic.epoch] = lrs

        to_save = self.decide_to_save()
        
        if to_save:
            data.save_step(
                self.model, self.data_loader.vocab_encoder,
                self.optimizer_m, self.opt,
                self.opt.train.dynamic.epoch, self.lrs_m, "story")
        
        for i, param_group in enumerate(self.optimizer_k.param_groups):
            lrs[i] = param_group['lr']
        self.lrs_k[self.opt.train.dynamic.epoch] = lrs
        if to_save:
            data.save_step(
                self.model_knowledge, self.data_loader.vocab_encoder,
                self.optimizer_k, self.opt,
                self.opt.train.dynamic.epoch, self.lrs_k, "knowledge")
        
    def log_losses(self, opt, losses):
        if (not cfg.toy and cfg.save) or cfg.test_save:
            data.save_eval_file(opt, losses["train"], "losses", split="train")
            data.save_eval_file(opt, losses['dev'], "losses", split="dev")
            data.save_eval_file(opt, losses['test'], "losses", split="test")

    def set_logger(self):
        if cfg.toy:
            self.logger = SummaryWriter(utils.make_name(
                self.opt, prefix="garbage/logs/", eval_=True, do_epoch=False))
        else:
            self.logger = SummaryWriter(utils.make_name(
                self.opt, prefix="logs/", eval_=True, do_epoch=False))
        print("Logging Tensorboard Files at: {}".format(self.logger.logdir))

    def stop_logger(self):
        self.logger.close()

    def run(self):
        self.set_logger()
        self.count = 0
        for epoch in range(self.epochs):
            self.model.train()
            self.model_knowledge.train()
            self.opt.train.dynamic.epoch += 1
            self.epoch()

        self.stop_logger()

    def epoch(self):
        nums_k, nums_s = self.reset_losses()

        # Initialize progress bar
        bar = utils.initialize_progress_bar(
            self.data_loader.sequences["train"])

        reset = False

        while not reset:
            loss, nums_k, nums_s, reset = self.do_forward_pass(nums_k, nums_s)
            self.do_backward_pass(loss)
            self.update_parameters()

            bar.update(self.opt.train.dynamic.bs)
            self.count += 1

            for loss_name in self.losses_k["train"]:
                self.logger.add_scalar(
                    "train/{}".format(loss_name),
                    loss.item() / self.opt.train.dynamic.bs,
                    self.count)
            
            for loss_name in self.losses_s["train"]:
                self.logger.add_scalar(
                    "train/{}".format(loss_name),
                    loss.item() / self.opt.train.dynamic.bs,
                    self.count)

            if cfg.toy and self.counter(nums) > 300:
                break

        with torch.no_grad():
            self.run_evaluation_cycle()

        self.log_losses(self.opt, self.losses_k)
        self.log_losses(self.opt, self.losses_s)
        self.update_top_score(self.opt)
        self.save_model(self.get_tracked_score())

        self.data_loader.reset_offsets("train")

    def run_evaluation_cycle(self):
        for split in ["dev", "test"]:
            self.evaluator.validate(
                self.opt.train.dynamic.epoch, split,
                self.losses_k[split], self.losses_s[split])
            if self.do_gen:
                gen.do_gen_run(
                    self.opt, self.generator, self.opt.train.dynamic.epoch, split,
                    self.losses_s[split])
            iter_num = self.opt.train.dynamic.epoch

            for loss_name in self.losses_k[split]:
                self.logger.add_scalar(
                    "{}/{}".format(split, loss_name),
                    self.losses_k[split][loss_name][iter_num],
                    iter_num)
            
            for loss_name in self.losses_s[split]:
                self.logger.add_scalar(
                    "{}/{}".format(split, loss_name),
                    self.losses_s[split][loss_name][iter_num],
                    iter_num)

    def clip_gradients(self):
        if self.opt.train.static.clip:
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), self.opt.train.static.clip)
            torch.nn.utils.clip_grad_norm_(
                self.model_knowledge.parameters(), self.opt.train.static.clip)
                
    def do_forward_pass(self, nums_k, nums_s):
        token_loss_knowledge, token_loss_sentence, nums_k , nums_s, reset = self.batch(
            self.opt, nums_k, nums_s, self.losses_k["train"], self.losses_s["train"],
            self.batch_variables)
        return token_loss_knowledge, token_loss_sentence, nums_k, nums_s, reset

    def do_backward_pass(self, loss, flag):
        if flag==True:
            loss.backward(retain_graph=True)
        else:
            loss.backward()

    def update_parameters_knowledge(self):
        self.optimizer_k.step()
        self.optimizer_k.zero_grad()

    def update_parameters_sentence(self):
        self.optimizer_m.step()
        self.optimizer_m.zero_grad()

    def reset_losses(self):
        loss_names_k = set([i.rstrip("maicro").rstrip("_") for
                          i in self.losses_k["train"].keys()])
        loss_names_s = set([i.rstrip("maicro").rstrip("_") for
                          i in self.losses_s["train"].keys()])
        
        return self.initialize_losses(list(loss_names_k)), self.initialize_losses(list(loss_names_s))


class IteratorTrainer(Trainer):
    def __init__(self, opt, meta, data_loader, model, model_knowledge, optimizer_m, optimizer_k):
        super(IteratorTrainer, self).__init__(
            opt, meta, data_loader, model, model_knowledge, optimizer_m, optimizer_k)

        self.iters = meta.cycle
        self.total_iters = meta.iterations

    def run(self):
        self.set_logger()

        # Initialize progress bar
        bar = utils.set_progress_bar(self.total_iters)

        for cycle_num in range(int(self.total_iters / self.iters)):
            self.model_knowledge.train() 
            self.model.train()
            self.cycle(bar, cycle_num)

            with torch.no_grad():
                self.run_evaluation_cycle()

            self.log_losses(self.opt, self.losses_k)
            self.update_top_score(self.opt)
        self.save_model(self.get_tracked_score())

        self.stop_logger()

    def cycle(self, bar, cycle_num):
        nums_k, nums_s = self.reset_losses()
        print("Nums", nums_k, nums_s)
        torch.autograd.set_detect_anomaly(True)
        for i in range(1, self.iters + 1):
            loss_k, loss_s, nums_k, nums_s, reset = self.do_forward_pass(nums_k, nums_s)
            
            self.optimizer_k.zero_grad()
            self.optimizer_m.zero_grad()
            self.do_backward_pass(loss_k, True)
            self.update_parameters_knowledge()
            self.do_backward_pass(loss_s, False)
            self.update_parameters_sentence()
            
            self.opt.train.dynamic.epoch += 1
            
            for loss_name in self.losses_k["train"]:
                self.logger.add_scalar(
                    "train/{}".format(loss_name),
                    loss_k.item() / self.opt.train.dynamic.bs,
                    self.opt.train.dynamic.epoch)
            for loss_name in self.losses_s["train"]:
                self.logger.add_scalar(
                    "train/{}".format(loss_name),
                    loss_s.item() / self.opt.train.dynamic.bs,
                    self.opt.train.dynamic.epoch)
            bar.update(1)
            
            if cfg.toy and i > 10:
                break

            if reset:
                self.data_loader.reset_offsets("train")


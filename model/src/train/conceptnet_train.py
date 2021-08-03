import random
import torch

import src.data.config as cfg

import src.train.atomic_train as base_train
import src.train.batch as batch_utils
import src.evaluate.conceptnet_evaluate as evaluate
import src.evaluate.conceptnet_generate as gen


def make_trainer(opt, *args):
    return ConceptNetGenerationIteratorTrainer(opt, *args)


class ConceptNetGenerationIteratorTrainer(
        base_train.AtomicGenerationIteratorTrainer):
    def set_evaluator(self, opt, model, model_knowledge, data_loader):
        self.evaluator = evaluate.make_evaluator(
            opt, model, model_knowledge, data_loader)

    def set_generator(self, opt, model, model_knowledge, data_loader):
        self.generator = gen.make_generator(
            opt, model, model_knowledge, data_loader)

    def batch(self, opt, *args):
        outputs = batch_utils.batch_atomic_generate(opt, *args)

        token_loss_knowledge = outputs["loss_knowledge"] #+ outputs["loss_sentence"]
        token_loss_sentence = outputs["loss_sentence"] #+ outputs["loss_knowledge"]
        nums_s = outputs["nums_s"]
        nums_k = outputs["nums_k"]
        reset = outputs["reset"]

        return token_loss_knowledge, token_loss_sentence, nums_k, nums_s, reset

    def update_top_score(self, opt):
        print(self.top_score)

        tracked_scores = self.get_tracked_score()

        if self.top_score is None:
            self.top_score = \
                self.top_score = {"epoch": {}, "score": {}}
            self.top_score["epoch"]["total_micro"] = self.opt.train.dynamic.epoch
            self.top_score["score"]["total_micro"] = tracked_scores["total_micro"]
        else:
            if tracked_scores["total_micro"] < self.top_score["score"]["total_micro"]:
                self.top_score["epoch"]["total_micro"] = self.opt.train.dynamic.epoch
                self.top_score["score"]["total_micro"] = tracked_scores["total_micro"]

        print(self.top_score)

    def get_tracked_score(self):
        return {
            "total_micro": self.losses_k["dev"]["total_micro"][self.opt.train.dynamic.epoch]
        }

    def decide_to_save(self):
        to_save = cfg.save and not cfg.toy

        curr_epoch = self.opt.train.dynamic.epoch

        to_save = to_save or cfg.test_save
        print(cfg.save_strategy)
        if cfg.save_strategy == "best":
            if ((self.top_score["epoch"]["total_micro"] != curr_epoch)):
                to_save = False
        to_save = True
        return to_save

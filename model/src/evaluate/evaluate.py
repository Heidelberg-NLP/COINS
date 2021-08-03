import time
import torch

import utils.utils as utils
import src.data.config as cfg


class Evaluator(object):
    def __init__(self, opt, model, model_knowledge, data_loader):
        super(Evaluator, self).__init__()

        self.data_loader = data_loader
        self.model = model
        self.model_knowledge = model_knowledge
        self.batch_variables = {
            "model": model,
            "model_knowledge": model_knowledge,
            "data": data_loader
        }

        self.opt = opt

    def validate(self, l, split="dev", losses_k={}, losses_s = {}, keyset=None):
        self.batch_variables["split"] = split
        print("Evaluating {}".format(split))

        epoch_losses_k, epoch_losses_s = self.epoch(
            self.opt, self.model, self.model_knowledge, self.data_loader, split, keyset)

        self.print_result(split, epoch_losses_k, epoch_losses_s)

        for loss_name, loss_val in epoch_losses_k.items():
            losses_k.setdefault(loss_name, {})
            losses_k[loss_name][l] = loss_val
        
        for loss_name, loss_val in epoch_losses_s.items():
            losses_s.setdefault(loss_name, {})
            losses_s[loss_name][l] = loss_val

    def epoch(self, opt, model, model_knowledge, data_loader, split, keyset=None):
        average_loss_k, average_loss_s, nums_k, nums_s = self.initialize_losses()
                
        data_loader.reset_offsets(splits=split, shuffle=False)

        # Set evaluation mode
        model.eval()
        model_knowledge.eval()

        start = time.time()

        # Initialize progress bar
        bar = utils.set_progress_bar(
            data_loader.total_size[split])

        reset = False

        with torch.no_grad():
            while not reset:

                start = data_loader.offset_summary(split)

                outputs = self.batch(
                    opt, nums_k, nums_s, average_loss_k, average_loss_s,
                    self.batch_variables, eval_mode=True)

                end = data_loader.offset_summary(split)

                reset = outputs["reset"]

                if not reset:
                    bar.update(end - start)
                else:
                    print(end)

                if cfg.toy and self.counter(nums) > 100:
                    break
                if (opt.eval.es != "full" and
                        (self.counter(nums) > opt.eval.es)):
                    break

        nums_k = outputs["nums_k"]
        nums_s = outputs["nums_s"]
        torch.cuda.synchronize()

        print("{} evaluation completed in: {} s".format(
            split.capitalize(), time.time() - start))

        average_loss_k = self.compute_final_scores(
            average_loss_k, nums_k)
        average_loss_s = self.compute_final_scores(
            average_loss_s, nums_s)

        return average_loss_k, average_loss_s 

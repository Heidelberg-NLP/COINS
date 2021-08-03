
import copy

import torch
import torch.nn as nn
import torch.nn.functional as F

import src.data.config as cfg
import src.train.utils as train_utils
import src.models.utils as model_utils
import src.evaluate.utils as eval_utils
import utils.utils as utils
from IPython import embed


##############################################################################
#                                       BATCH
##############################################################################


def batch_atomic_generate(opt, nums_k, nums_s, losses_k, losses_s, batch_variables, eval_mode=False):
    data_loader = batch_variables["data"]
    model = batch_variables["model"]
    model_knowledge = batch_variables["model_knowledge"]
    split = batch_variables["split"]

    batch, reset = data_loader.sample_batch(split, bs=opt.train.dynamic.bs)
    # Set loss name
    micro_name = "total_micro"
    macro_name = "total_macro"
    
    input_ = batch["sequences"]
    attention_mask = batch["attention_mask"]
    loss_mask = batch["loss_mask"]
    bs = input_.size(0)
    
    for i in range(2):
        current_input_knowledge = input_[:, 0, i, :].unsqueeze(-1)
        current_input_story_completion = input_[:, 1, i, :].unsqueeze(-1)
        #targets
        targets_knowledge = current_input_knowledge.squeeze(0)[:, 1:, 0].contiguous().view(-1)
        targets_story_completion = current_input_story_completion.squeeze(0)[:, 1:, 0].contiguous().view(-1)     
        #attention
        attention_mask_knowledge = attention_mask[:, 0, i, :-1]
        attention_mask_sentence = attention_mask[:, 1, i, :-1]     
        
         
        loss_know, _ = mle_steps(opt.net.model, model_knowledge, current_input_knowledge[:, :-1, :], targets_knowledge, attention_mask_knowledge, loss_reduction="none")
        loss_sentence, _ = mle_steps(opt.net.model, model, current_input_story_completion[:, :-1, :], targets_story_completion, attention_mask_sentence, loss_reduction="none")
                
        length_know = loss_mask[:, 0, i, :].sum(1)
        length_sentence = loss_mask[:, 1, i, :].sum(1)

        temp_loss_know = (loss_know * loss_mask[:, 0, i, :]).sum(1) 
        temp_loss_sentence = (loss_sentence * loss_mask[:, 1, i, :]).sum(1) 
         
        update_generation_losses(losses_k, nums_k, micro_name, macro_name, bs,
                             length_know, temp_loss_know, split, "knowledge")
        
        update_generation_losses(losses_s, nums_s, micro_name, macro_name, bs,
                             length_sentence, temp_loss_sentence, split, "sentence")
        if i==0:
            final_loss_knowledge = temp_loss_know / length_know
            final_loss_sentence =  temp_loss_sentence / length_sentence
        else:
            final_loss_knowledge += temp_loss_know / length_know 
            final_loss_sentence += temp_loss_sentence / length_sentence
        
        
    outputs = {"loss_knowledge": final_loss_knowledge.sum(), "loss_sentence": final_loss_sentence.sum(), "nums_k": nums_k, "nums_s": nums_s, "reset": reset}
    return outputs


def batch_conceptnet_generate(opt, nums_k, nums_s, losses_k, losses_s,  batch_variables,
                              eval_mode=False, tracking_mode=False):

    data_loader = batch_variables["data"]
    model = batch_variables["model"]
    model_knowledge = batch_variables["model_knowledge"]
    split = batch_variables["split"]

    batch, reset = data_loader.sample_batch(split, bs=opt.train.dynamic.bs)
    # Set loss name
    micro_name = "total_micro"
    macro_name = "total_macro"
    
    input_ = batch["sequences"]
    attention_mask = batch["attention_mask"]
    loss_mask = batch["loss_mask"]
    bs = input_.size(0)

    for i in range(2):
        current_input_knowledge = input_[:, 0, i, :].unsqueeze(-1)
        current_input_story_completion = input_[:, 1, i, :].unsqueeze(-1)
        #targets
        targets_knowledge = current_input_knowledge.squeeze(0)[:, 1:, 0].contiguous().view(-1)
        targets_story_completion = current_input_story_completion.squeeze(0)[:, 1:, 0].contiguous().view(-1)     
        #attention
        attention_mask_knowledge = attention_mask[:, 0, i, :-1]
        attention_mask_sentence = attention_mask[:, 1, i, :-1]     
        
         
        loss_know, _ = mle_steps(opt.net.model, model_knowledge, current_input_knowledge[:, :-1, :], targets_knowledge, attention_mask_knowledge, loss_reduction="none")
        loss_sentence, _ = mle_steps(opt.net.model, model, current_input_story_completion[:, :-1, :], targets_story_completion, attention_mask_sentence, loss_reduction="none")
                
        length_know = loss_mask[:, 0, i, :].sum(1)
        length_sentence = loss_mask[:, 1, i, :].sum(1)

        temp_loss_know = (loss_know * loss_mask[:, 0, i, :]).sum(1) 
        temp_loss_sentence = (loss_sentence * loss_mask[:, 1, i, :]).sum(1) 
        
        update_generation_losses(losses_k, nums_k, micro_name, macro_name, bs,
                             length_know, temp_loss_know, split, "knowledge")
        
        update_generation_losses(losses_s, nums_s, micro_name, macro_name, bs,
                             length_sentence, temp_loss_sentence, split, "sentence")
        
        if i==0:
            final_loss_knowledge = temp_loss_know / length_know
            final_loss_sentence =  temp_loss_sentence / length_sentence
        else:
            final_loss_knowledge += temp_loss_know / length_know 
            final_loss_sentence += temp_loss_sentence / length_sentence
        
    outputs = {"loss_knowledge": final_loss_knowledge.sum(), "loss_sentence": final_loss_sentence.sum(), "nums_k": nums_k, "nums_s": nums_s, "reset": reset}
        
    if tracking_mode:
        outputs["tracking"] = final_loss_knowledge.squeeze().tolist()
    return outputs


def mle_steps(key, model, input_, targets, attention_mask,
              loss_reduction="mean", i=None):
     
    word_acts = decode(model, input_.unsqueeze(1),
                      attention_mask, i)

    word_dist = train_utils.modify_output_for_loss_fn(
        "nll", word_acts, dim=-1)
    #print(word_dist.size())
    #print(word_dist.view(-1, word_dist.size(-1)).size())
    #print(targets.size())
    
    loss = F.nll_loss(
        word_dist.view(-1, word_dist.size(-1)),
        targets, reduction=loss_reduction)
    
    if loss_reduction != "mean":
        return loss.view(word_dist.size(0), -1), word_dist
    else:
        return loss, word_dist



def decode(model, input_, attention_mask, i):
    return model(input_ids = input_, attention_mask=attention_mask)


def update_generation_losses(losses, nums, micro, macro, bs,
                             length, loss, split, types):
    if split == "train":
        train_utils.update_generation_losses(
            losses, nums, micro, macro, bs, length, loss, types)
    else:
        eval_utils.update_generation_losses(
            losses, nums, micro, macro, bs, length, loss, types)

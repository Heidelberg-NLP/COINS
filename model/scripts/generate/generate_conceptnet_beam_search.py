import os
import time
import sys
import argparse
sys.path.append(os.getcwd())
import torch

import src.train.atomic_train as train
import src.models.models as models
import src.models.model_knowledge_story as model_knowledge_story
import src.data.data as data
import utils.utils as utils
import src.train.utils as train_utils
import src.data.config as cfg

from src.data.utils import TextEncoder
from transformers import GPT2Tokenizer
from src.train.opt import OpenAIAdam

import src.models.utils as model_utils
from tqdm import tqdm
import torch.nn.functional as F
import numpy as np
import random
from torch.nn import CrossEntropyLoss

parser = argparse.ArgumentParser()
parser.add_argument("--generation_set_size", type=str, default='full', choices=["full", "human"])
parser.add_argument("--device", type=int, default=0)
parser.add_argument("--split", type=str, default="dev")
parser.add_argument("--beam", type=int, default=10)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--experiment_num", type=str, default="0")
parser.add_argument("--path", type=str, default="")
parser.add_argument("--model_name", type=str, default="models/conceptnet-generation/iteration-500-100000/transformer/rel_language-trainsize_100-devversion_12-maxe1_10-maxe2_15/model_transformer-nL_12-nH_12-hSize_768-edpt_0.1-adpt_0.1-rdpt_0.1-odpt_0.1-pt_gpt-afn_gelu-npos_1-demb_F-init_pt-vSize_40545/exp_generation-seed_123-es_0-l2_0.01-vl2_T-lrsched_warmup_linear-lrwarm_0.002-clip_1-loss_nll-b2_0.999-b1_0.9-e_1e-08/bs_1-trick_0-smax_40-sample_beam-numseq_1-gs_full-es_full-categories_None/1e-05_adam_64_13500.pickle")
parser.add_argument("--model_knowledge_name", type=str, default="models/conceptnet-generation/iteration-500-100000/transformer/rel_language-trainsize_100-devversion_12-maxe1_10-maxe2_15/model_transformer-nL_12-nH_12-hSize_768-edpt_0.1-adpt_0.1-rdpt_0.1-odpt_0.1-pt_gpt-afn_gelu-npos_1-demb_F-init_pt-vSize_40545/exp_generation-seed_123-es_0-l2_0.01-vl2_T-lrsched_warmup_linear-lrwarm_0.002-clip_1-loss_nll-b2_0.999-b1_0.9-e_1e-08/bs_1-trick_0-smax_40-sample_beam-numseq_1-gs_full-es_full-categories_None/1e-05_adam_64_13500.pickle")
parser.add_argument("--gen_len", type=int, default=250)

args = parser.parse_args()
split = args.split

# Generate configuration files depending on experiment being run
#utils.generate_config_files("conceptnet", args.experiment_num, eval_mode=True)

# Loads the correct configuration file
config_file = "config/conceptnet/config_{}.json".format(args.experiment_num)

# Read config file to option
config = cfg.read_config(cfg.load_config(config_file))
cfg.device = args.device
eval_opt = cfg.get_eval_parameters(config)

model_stuff = data.load_checkpoint(args.model_name)
model_know_stuff = data.load_checkpoint(args.model_knowledge_name)


opt = model_stuff["opt"]
opt.eval.update(eval_opt)

# Set the random seeds
torch.manual_seed(opt.train.static.seed)
random.seed(opt.train.static.seed)
if config.gpu_mode:
    torch.cuda.manual_seed_all(opt.train.static.seed)

# Where to find the data
splits = ["train", "dev", "test"]

opt.train.dynamic.epoch = 0

print("Loading Data")

if "maxr" not in opt.data.keys():
    opt.data.maxr = 5 if opt.data.rel == "language" else 1


x = "data/conceptnet/processed/generation/rel_language-trainsize_100-devversion_12-maxe1_200-maxe2_200.pickle"

path= x.format(
    utils.make_name_string(opt.data))

data_loader = data.make_data_loader(opt)
loaded = data_loader.load_data(path)

data_loader.opt = opt
data_loader.batch_size = opt.train.dynamic.bs

print("Done.")

text_encoder = GPT2Tokenizer.from_pretrained('gpt2')
special_tokens = {"cls_token":"[CLS]", "unk_token":"[UNK]"}
text_encoder = GPT2Tokenizer.from_pretrained("gpt2", cls_token="[CLS]", unk_token="[UNK]", mask= '["MASK"]', separator='["SEP"]', start_of_sentence='["SOS"]', end_of_sentence='["EOS"]')
text_encoder.add_special_tokens(special_tokens)
    
#categories = data.conceptnet_data.conceptnet_relations
    
special = [data.start_token, data.end_token]
#special += ["<{}>".format(cat) for cat in categories]

start_token = "<START>"
end_token = "<END>"

categories =  [
    'AtLocation', 'CapableOf', 'Causes', 'CausesDesire', 'CreatedBy',
    'DefinedAs', 'DesireOf', 'Desires', 'HasA', 'HasFirstSubevent',
    'HasLastSubevent', 'HasPainCharacter', 'HasPainIntensity',
    'HasPrerequisite', 'HasProperty', 'HasSubevent', 'InheritsFrom',
    'InstanceOf', 'IsA', 'LocatedNear', 'LocationOfAction', 'MadeOf',
    'MotivatedByGoal', 'NotCapableOf', 'NotDesires', 'NotHasA',
    'NotHasProperty', 'NotIsA', 'NotMadeOf', 'PartOf', 'ReceivesAction',
    'RelatedTo', 'SymbolOf', 'UsedFor', 'OneCounterfactCanbe', 'TheReasonsare'
]
special = [start_token, end_token]
# if opt.data.rel == "relation":
#special += ["<{}>".format(cat) for cat in categories]

text_encoder.encoder = data_loader.vocab_encoder
text_encoder.decoder = data_loader.vocab_decoder


context_size_i1 = data_loader.max_input1
context_size_i2 = data_loader.max_input2
context_size_i3 = data_loader.max_input3
context_size_i4 = data_loader.max_input4
context_size_o1 = data_loader.max_output1
context_size_o2 = data_loader.max_output2
context_size_o3 = data_loader.max_output3
context_size_o4 = data_loader.max_output4
    
    #opt.data.maxr = context_size_r

n_special = len(special)
n_ctx = context_size_i1 + context_size_i2 + context_size_i3 + context_size_i4 + context_size_o1 + context_size_o2 + context_size_o3 + context_size_o4
n_vocab = len(text_encoder.encoder) + n_ctx

print(data_loader.__dict__.keys())
opt.net.vSize = n_vocab

print("Building Model")

print(opt.exp)

model = models.make_model(opt, n_vocab, n_ctx, 0, load=False, return_acts=True, return_probs=False)
model.resize_token_embeddings(len(text_encoder))
models.load_state_dict(model, model_stuff["state_dict"])


model_know = model_knowledge_story.make_model(opt, n_vocab, n_ctx, 0, load=False, return_acts=True, return_probs=False)
model_know.resize_token_embeddings(len(text_encoder))
model_knowledge_story.load_state_dict(model_know, model_know_stuff["state_dict"])


if config.gpu_mode:
    print("Pushing to GPU: {}".format(config.gpu_index))
    cfg.device = config.gpu_index
    cfg.do_gpu = True
    torch.cuda.set_device(cfg.device)
    model.cuda(cfg.device)
    model_know.cuda(cfg.device)
    print("Done.")

model.eval()
model_know.eval()

device = cfg.device
model.to(device)
model_know.to(device)

random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)


def make_batch(X):
    X = np.array(X)
    assert X.ndim in [1, 2]
    if X.ndim == 1:
        X = np.expand_dims(X, axis=0)
    pos_enc = np.arange(n_vocab + n_special, n_vocab + n_special + X.shape[-1])
    pos_enc = np.expand_dims(pos_enc, axis=0)
    batch = np.stack([X, pos_enc], axis=-1)
    batch = torch.tensor(batch, dtype=torch.long).to(device)
    return batch


def append_batch(X, beam_toks, mask):
    beam_toks = beam_toks.unsqueeze(1)
    next_x = beam_toks.unsqueeze(1)
    next_mask = torch.cat([mask, torch.ones(X.size(0), 1, device=mask.device)], 1)
    return torch.cat((X, next_x), 1), next_mask
    
    
def beam_generate_sequence(XMB, MMB, lm_model, size_end, k):

             tokens = []
             beam_losses = []
             loss_fct = CrossEntropyLoss(reduction="none")
             XMB = XMB.unsqueeze(-1)
             # Beam Search
             beam_lls, beam_toks, beam_seqs = None, None, None
             lm_probs = F.log_softmax(lm_model(XMB.unsqueeze(1), attention_mask=MMB), dim=-1)
             dist = lm_probs[:, -1, :].squeeze()
             values, indices = lm_probs[:,-1, :].max(dim=-1)
             seqs = indices.clone().unsqueeze(1)
             beam_lls, beam_toks = dist.topk(k)
             beam_losses.append(beam_lls)
             ended = (beam_toks == end_token).float()
             counts = (2 - ended)
             beam_toks = beam_toks.unsqueeze(1)
             beam_seqs = beam_toks.clone()
             XMB = XMB.repeat(k, 1, 1)
             MMB = MMB.repeat(k, 1)
             next_pos = XMB[:, 1, -1:] + 1
              
             next_x = beam_toks.unsqueeze(1)
             XMB = torch.cat((XMB, next_x), 1)
             MMB = torch.cat([MMB, torch.ones(XMB.size(0), 1, device=MMB.device)], 1)
             # Compute distribution for current beam
             for _ in range(args.gen_len):
                   lm_probs = F.log_softmax(lm_model(XMB.unsqueeze(1), attention_mask=MMB), dim=-1)
                   dist = lm_probs[:, -1, :].squeeze()
             # get hypothesis tokens for distribution
                   hyp_beam_lls, hyp_beam_toks = dist.topk(k)
            
             # Compute masks and expand beam
                   expanded_ended = ended.unsqueeze(1).repeat(1, k)
                   hypothesis_mask = expanded_ended * kill_mask + (1 - expanded_ended)
                   current_beam_lls = beam_lls.unsqueeze(1).repeat(1, k).view(k**2)

             # Compute losses of hypotheses, masking those that have ended
                   hyp_beam_lls = (hyp_beam_lls.view(k**2) *
                            hypothesis_mask.view(-1)) + current_beam_lls
             # Get normalizer for sequences
                   temp_counts = counts.unsqueeze(1).repeat(1, k).view(k ** 2)

             # Select best beams with lowest aggregate loss
                   beam_lls, top_beam_idxs = (hyp_beam_lls / temp_counts).topk(k)
             # Update placements in beam based on selecetion
                   beam_losses = [i.index_select(0, top_beam_idxs // k)
                           for i in beam_losses]
                   ended = ended.index_select(0, top_beam_idxs // k)
                   counts = temp_counts.index_select(0, top_beam_idxs)

             # Save beam losses
                   beam_losses.append(beam_lls * counts)
             # Update beam tokens
                   ended_mask = (1 - ended).long()
                   end_replacement = (end_token * ended).long()
                   next_toks = hyp_beam_toks.view(-1)[top_beam_idxs]
                   beam_toks = next_toks * ended_mask + end_replacement

             # Update ended and counts
                   ended = ended + (beam_toks == end_token).float() * (1 - ended)
                   counts = counts + (1 - ended)

             # Update beam sequences
                   if (beam_toks !=end_token).sum().item():
                       beam_seqs = beam_seqs.t().repeat(k, 1).t().contiguous().view(k**2, -1)[top_beam_idxs]
                       beam_seqs = torch.cat((beam_seqs, beam_toks.unsqueeze(1)), dim=1)
            
             # I have no idea what's going on but Ari's on point with it
                   XMB = XMB.transpose(0, 1).transpose(1, 2).repeat(k, 1, 1).transpose(2, 1).transpose(1, 0).contiguous().view(k**2, XMB.size(1), XMB.size(2))[top_beam_idxs]
            
                   XMB, MMB = append_batch(XMB, beam_toks, MMB)

                   if (beam_toks == end_token).sum().item() == k or _ == size_end - 1:
                          break
                                
             beams = []
             value = 50261
             beam_seqs = beam_seqs[beam_seqs!=value]
             beam_seqs = beam_seqs.unsqueeze(0)
             
             value = 50260
             beam_seqs = beam_seqs[beam_seqs!=value]
             beam_seqs = beam_seqs.unsqueeze(0)
             
             for beam in beam_seqs:
                          beams.append(" ".join("".join([text_encoder.decoder[tok.item()].replace(
                    '</w>', ' ').replace('\n', '').replace('Ġ', ' ')
                     for tok in beam if tok != end_token]).split()))
             sampling_result = {
                 "sequence": beams[0],
                 "beams": beams,
                 "beam_lengths": [counts],
                 "length": counts
             }
             return sampling_result, beam_seqs

def greedy_append_batch(X, next_idx, mask):
        
        next_x = next_idx.unsqueeze(1) #torch.cat((next_idx, next_pos), -1).unsqueeze(1)
        next_mask = torch.cat([mask, torch.ones(X.size(0), 1, device=mask.device)], 1)
        return torch.cat((X, next_x), 1), next_mask

def greedy_generate_sequence(XMB, MMB, lm_model, end_len):
        XMB = XMB.unsqueeze(-1)

        lm_probs = F.log_softmax(lm_model(
            XMB.unsqueeze(1), attention_mask=MMB), dim=-1)

        values, indices = lm_probs[:,-1, :].max(dim=-1)
        seqs = indices.clone().unsqueeze(1)

        loss = values
        counts = 1
        XMB = torch.cat((XMB, indices.view(-1, 1).unsqueeze(1)), 1)
        MMB = torch.cat([MMB, torch.ones(XMB.size(0), 1, device=MMB.device)], 1)
        # Sample from top k
        for _ in range(args.gen_len):
            lm_probs = F.log_softmax(lm_model(
                XMB.unsqueeze(1), attention_mask=MMB), dim=-1)
            
            # Sample from top k
            values, next_idx = lm_probs[:, -1, :].max(dim=-1)
            loss += values
            counts += 1

            next_idx = next_idx.unsqueeze(1)

            seqs = torch.cat([seqs, next_idx], 1)

            if (next_idx.item() == end_token) or (_ == end_len - 1):
                break

            XMB, MMB = greedy_append_batch(XMB, next_idx, MMB)

        beams = []

        for beam in seqs:

            beams.append(" ".join("".join(
                [data_loader.vocab_decoder[tok.item()].replace(
                    '</w>', ' ').replace('\n', '')
                 for tok in beam if tok != end_token]).split()))
        
        sampling_result = {
            "sequence": " ".join(beams),
            "beams": beams,
            "beam_losses": [loss.item()],
            "loss": loss.item(),
            "beam_lengths": [counts],
            "length": counts
        }
        return sampling_result, seqs

data_loader.reset_offsets(splits=split, shuffle=False)

# Generate for all sequences
b = [tuple(j) for j in data_loader.sequences[split]['total'][:, :data_loader.max_input_len_k].tolist()]
total = list(range(len(b)))

args.decoding_strategy = "beam"

kill_mask = torch.ones(args.beam, args.beam).to(device) * 9000
kill_mask[:, 0] = 0

final_sequences = []

end_token = text_encoder.encoder["<END>"]

# Make Eval File name -- can probably comment this if you're doing your
# own file naming convention. I only include this to keep it consistent
eval_file_name = args.model_name.replace("sample_greedy", "sample_{}".format(
    args.decoding_strategy))

eval_file_name = eval_file_name.replace("bs_1", "bs_{}".format(args.beam))
eval_file_name = eval_file_name[:-7] + "/{}.pickle".format(split)
eval_file_name = eval_file_name.replace("models/", "results/gens/")

print("Saving generations to: {}".format(eval_file_name))

output_s = []
loss = 0.0
generate_counter = 0
target_counter = 0 
count =0 

start_idx_k = data_loader.max_input_len_k 
max_end_len_k = data_loader.max_output_len_k
start_idx_s = data_loader.max_input_len_s
max_end_len_s = data_loader.max_output_len_s

print(start_idx_k)
print(max_end_len_k)
print(start_idx_s)
print(max_end_len_s)


#print(text_encoder)


with torch.no_grad():
    for idx in tqdm(total):
        sequence_all = {}
        
        output_sentence = ""
        output_knowledge = ""

        batch, reset = data_loader.sample_batch(split=split, bs=1, idxs=[idx])
        input_ = batch["sequences"]
        attention_mask = batch["attention_mask"]
        knowledge_hashtag1 = torch.LongTensor(text_encoder.encode(' # Effect # ')).to(cfg.device)
        knowledge_hashtag2 = torch.LongTensor(text_encoder.encode(' # Cause # ')).to(cfg.device)
        
        sentence_hashtag1 = torch.LongTensor(text_encoder.encode(' # 1 Next Sentence # ')).to(cfg.device)
        sentence_hashtag2 = torch.LongTensor(text_encoder.encode(' # 2 Next Sentence # ')).to(cfg.device)
        mask_token = torch.LongTensor(text_encoder.encode('["MASK"]')).to(cfg.device)
        
        for i in range(2):
            print("Iteration....", i)
            if i==0:
                # Prepare data for Knowledge Generation
                XMB_knowledge = input_[:, 0, i, :start_idx_k] 
                MMB_knowledge = attention_mask[:, 0, i, :start_idx_k]
                second_context = " ".join([text_encoder.decoder[i].replace('</w>', ' ').replace(
                                    "<blank>", "___ ") for i in XMB_knowledge[:, :].squeeze().tolist() if i])
                
                # Decode Knowledge
                sampling_result, XMB = beam_generate_sequence(XMB_knowledge, MMB_knowledge, model_know, max_end_len_k, 1)
                output_knowledge = sampling_result["sequence"]

                #print(output_knowledge)
                 
                # Prepare data for Knowledge Generation
                sentence12_id = input_[:, 2, 1, :start_idx_s]

                XMB = torch.cat((XMB, sentence_hashtag1.unsqueeze(0)),1)
                XMB = torch.cat((XMB, sentence12_id), 1)
                XMB = sentence12_id
                XMB = XMB[:, :start_idx_s]
                MMB = (XMB!= 0).float().to(cfg.device)
                second_context = " ".join([text_encoder.decoder[i].replace('</w>', ' ').replace(
                                    "<blank>", "___ ") for i in XMB[:, :].squeeze().tolist() if i])
                #print("Input2", second_context.replace('Ġ', ' '))
                # Decode Sentence                                
                sampling_result, XMB_knowledge = beam_generate_sequence(XMB, MMB, model, max_end_len_s, 1)
                previous_XMB_knowledge = XMB_knowledge
                 
                output_sentence = sampling_result["sequence"]

                #print(output_sentence)
                #exit()

            elif i>0:
                # Prepare data for Knowledge Generation
                initial_context_id = input_[:, 2, 1, :]
                #print(initial_context_id)
                second_context = " ".join([text_encoder.decoder[i].replace('</w>', ' ').replace(
                                    "<blank>", "___ ") for i in initial_context_id[:, :].squeeze().tolist() if i])
                #print("First two sentences", second_context.replace('Ġ', ' '))
                initial_context_id = initial_context_id.squeeze()[torch.nonzero(initial_context_id.squeeze(0))].squeeze().unsqueeze(0)      
                second_context = " ".join([text_encoder.decoder[i].replace('</w>', ' ').replace(
                                    "<blank>", "___ ") for i in initial_context_id[:, :].squeeze().tolist() if i])
                
                #print("First two sentences", second_context.replace('Ġ', ' '))
                
                XMB_knowledge_ = torch.LongTensor(1, start_idx_k).fill_(0).to(cfg.device)
                XMB_knowledge = torch.cat((initial_context_id, previous_XMB_knowledge), 1)
                XMB_knowledge = torch.cat((XMB_knowledge, mask_token.unsqueeze(0)), 1)
                
                sentence_5_id = input_[:, 2, 2, :start_idx_k]
                sentence_5_id = sentence_5_id.squeeze()[torch.nonzero(sentence_5_id.squeeze(0))].squeeze().unsqueeze(0) 
                XMB_knowledge = torch.cat((XMB_knowledge, sentence_5_id),1)
                XMB_knowledge = torch.cat((XMB_knowledge, knowledge_hashtag2.unsqueeze(0)),1)
                XMB_knowledge = torch.cat((XMB_knowledge, sentence_5_id),1)
                
                second_context = " ".join([text_encoder.decoder[i].replace('</w>', ' ').replace(
                                    "<blank>", "___ ") for i in XMB_knowledge[:, :].squeeze().tolist() if i])
                
                #print("Second input for knowledge", second_context.replace('Ġ', ' '))
                
                 
                XMB_knowledge_[:, :XMB_knowledge.size(1)] = XMB_knowledge                

                MMB_knowledge  = (XMB_knowledge_!= 0).float().to(cfg.device)
                second_context = " ".join([text_encoder.decoder[i].replace('</w>', ' ').replace(
                                    "<blank>", "___ ") for i in XMB_knowledge_[:, :].squeeze().tolist() if i])
                #print(second_context.replace('Ġ', ' '))
                
                # Decode Knowledge
                sampling_result, XMB = beam_generate_sequence(XMB_knowledge_, MMB_knowledge, model_know, max_end_len_k, 1)
                output_knowledge = output_knowledge +' ["Second Knowledge"] '+sampling_result["sequence"]
                
                # Prepare data for Knowledge Generation
                sentence_2_id = input_[:, 2, 0, :]
                sentence_2_id = sentence_2_id.squeeze()[torch.nonzero(sentence_2_id.squeeze(0))].squeeze().unsqueeze(0) 
                XMB_ = torch.LongTensor(1, start_idx_s+max_end_len_s).fill_(0).to(cfg.device)
                XMB = torch.cat((XMB, sentence_hashtag2.unsqueeze(0)),1)
                XMB = torch.cat((XMB, sentence_2_id),1)
                XMB = torch.cat((XMB, previous_XMB_knowledge), 1)
                #XMB = XMB[:, :start_idx_s]
                XMB = torch.cat((sentence_2_id, previous_XMB_knowledge), 1)
                XMB_[:, :XMB.size(1)]= XMB
                second_context = " ".join([text_encoder.decoder[i].replace('</w>', ' ').replace(
                                    "<blank>", "___ ") for i in XMB[:, :].squeeze().tolist() if i])
                #print("Second input for knowledge", second_context.replace('Ġ', ' '))
                
                XMB_ = XMB_[:, :start_idx_s]
                MMB = (XMB_!= 0).float().to(cfg.device)
                
                # Decode Sentence                                
                sampling_result, XMB_knowledge = beam_generate_sequence(XMB_, MMB, model, max_end_len_s, 1)
                output_sentence = output_sentence +' '+sampling_result["sequence"]
                                
                f = open("sentence_"+args.path+".txt", "a")
                f.write(str(output_sentence)+'\n')
                f.close()
                f = open("knowledge_"+args.path+".txt", "a")
                f.write(str(output_knowledge)+'\n')
                f.close()
                output_sentence = ""
                output_knowledge = ""
                #exit()
                
                


import pickle

utils.mkpath("/".join(eval_file_name.split("/")[:-1]))

with open(eval_file_name, "wb") as f:
    pickle.dump(final_sequences, f)


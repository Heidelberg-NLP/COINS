import src.data.utils as data_utils
import src.data.atomic as adata
import src.data.config as cfg

import torch
import random
from tqdm import tqdm
import numpy as np


def map_name(name, opt):
    if name == "train":
        return "train{}k.txt".format(opt.trainsize)
    elif name == "test":
        return "test.txt"
    else:
        return "dev{}.txt".format(opt.devversion)


class GenerationDataLoader(adata.DataLoader):
    def __init__(self, opt, categories=None):
        super(GenerationDataLoader, self).__init__(opt)
        self.opt = opt

        for split in self.data:
            self.data[split] = {"total": []}
            self.offsets[split] = {"total": 0}

        self.vocab_encoder = None
        self.vocab_decoder = None
        self.special_chars = None
        
        self.max_input1 = None
        self.max_input2 = None
        self.max_input3 = None
        self.max_input4 = None
                
        self.max_output1 = None
        self.max_output2 = None
        self.max_output3 = None
        self.max_output4 = None
        self.max_input_len_k = None
        self.max_input_len_s = None
        self.max_output_len_k = None
        self.max_output_len_s = None

    def offset_summary(self, split):
        return sum(self.offsets[split].values())

    def load_data(self, path):
        if ".pickle" in path:
            print("Loading data from: {}".format(path))
            data_utils.load_existing_data_loader(self, path)
            return True

        for split in self.data:
            file_name = map_name(split, self.opt.data)
            if split != "dev" or self.opt.data.devversion != "12":
                string_tuples = open("{}/{}".format(
                    path, file_name), "r").read().split("\n")
                tuples = [x.split("\t") for x in string_tuples if x]
            elif split == "dev":
                string_tuples = open("{}/{}".format(path, "dev1.txt"), "r").read().split("\n")
                tuples = [x.split("\t") for x in string_tuples if x]
            

            if split in ["dev", "test"]:
                self.data[split]["total"] = \
                        [(i[0].lower().strip(), i[1].lower().strip(), i[2].lower().strip(), i[3].lower().strip(), i[4].lower().strip(), i[5].lower().strip(), i[6].lower().strip(), i[7].lower().strip(), i[8].lower().strip(), i[9].lower().strip(), i[10].lower().strip()      ) for i in tuples]
            else:
                self.data[split]["total"] = \
                        [(i[0].lower().strip(), i[1].lower().strip(), i[2].lower().strip(), i[3].lower().strip(), i[4].lower().strip(), i[5].lower().strip(), i[6].lower().strip(), i[7].lower().strip(), i[8].lower().strip(), i[9].lower().strip(), i[10].lower().strip()) for i in tuples]
            
        return False

    def make_tensors(self, text_encoder, special,
                     splits=["train", "dev", "test"], test=False):

        self.vocab_encoder = text_encoder.encoder
        self.vocab_decoder = text_encoder.decoder
        self.special_chars = special

        sequences = {}
        for split in splits:
            sequences[split] = get_generation_sequences(
                self.data, split, text_encoder, test, self.opt.data.maxe1,
                self.opt.data.maxe2)
          
            if split == "train":
                self.data[split]["total"] = [j for i, j in enumerate(
                    self.data[split]["total"])]
            
            self.masks[split]["total"] = [(len(i[0]), len(i[1]), len(i[2]), len(i[3]), len(i[4]), len(i[5]), len(i[6]), len(i[7]), len(i[8]), len(i[9]), len([10])) for i in sequences[split]]
            '''Incomplete Story(i.e, S1, S2 [SEP] S5) #Effect# S2 \t  **Ouput_Effect_S2** \t Incomplete Story(i.e, S1, S2 [SEP] S5) #Cause# S5 \t  **Ouput_Cause_S5** \t Incomplete Story(i.e, S1, S2 [SEP] S5) \t Incomplete Story(i.e, S1, S2 [SEP] S5) [SEP] Ouput_Effect_S2 [SEP] Ouput_Cause_S5 \t **Output_S3** \t Incomplete Story(i.e, S1, S2 S3 [SEP] S5) #Effect# S3 \t  **Ouput_Effect_S3** \t Incomplete Story(i.e, S1, S2 S3 [SEP] S5) #Cause# S5 \t  **Ouput_Cause_S5** \t Incomplete Story(i.e, S1, S2 S3 [SEP] S5) \t Incomplete Story(i.e, S1, S2 S3 [SEP] S5) [SEP] Ouput_Effect_S3 [SEP] Ouput_Cause_S5 \t **Output_S4** \t S2 +'\t'+ S1 +' '+ S2 +'\t'+ S5+ '\n''''
        self.max_input1 = max([max([l[0] for l in self.masks[split]["total"]])
                           for split in self.masks])
        self.max_input2 = max([max([l[2] for l in self.masks[split]["total"]])
                           for split in self.masks])
        self.max_input3 = max([max([l[4] for l in self.masks[split]["total"]])
                           for split in self.masks])
        self.max_input4 = max([max([l[6] for l in self.masks[split]["total"]])
                           for split in self.masks])
                   
        self.max_output1 = max([max([l[1] for l in self.masks[split]["total"]])
                           for split in self.masks])                      
        self.max_output2 = max([max([l[3] for l in self.masks[split]["total"]])
                           for split in self.masks])                    
        self.max_output3 = max([max([l[5] for l in self.masks[split]["total"]])
                           for split in self.masks])
        self.max_output4 = max([max([l[7] for l in self.masks[split]["total"]])
                           for split in self.masks])
        
        self.max_sentence2 = max([max([l[8] for l in self.masks[split]["total"]])
                           for split in self.masks])
        self.max_context = max([max([l[9] for l in self.masks[split]["total"]])
                           for split in self.masks])
        self.max_context = max([max([l[10] for l in self.masks[split]["total"]])
                           for split in self.masks])
        self.max_input_len_k = max(self.max_input1, self.max_input3)
        self.max_output_len_k = max(self.max_output1, self.max_output3) 
        

        self.max_input_len_s = max(self.max_input2, self.max_input4)
        self.max_output_len_s = max(self.max_output2, self.max_output4) 

         
        for split in splits:
            num_elements = len(sequences[split])
            self.sequences[split]["total"] = torch.LongTensor(num_elements, 3, 3, max(self.max_input_len_k + self.max_output_len_k, self.max_input_len_s + self.max_output_len_s)).fill_(0)
            print(self.sequences[split]["total"].size())
            print(split)
            for i, seq in enumerate(sequences[split]):
                                
                self.sequences[split]["total"][i, 0, 0, :len(seq[0])] = torch.LongTensor(seq[0]) 
                start = self.max_input_len_k 
                end = start + len(seq[1])
                self.sequences[split]["total"][i, 0, 0, start:end] = torch.LongTensor(seq[1])
                
                self.sequences[split]["total"][i, 0, 1, :len(seq[4])] = torch.LongTensor(seq[4])
                start = self.max_input_len_k 
                end = start + len(seq[5])
                self.sequences[split]["total"][i, 0, 1, start:end] = torch.LongTensor(seq[5])
                         
                self.sequences[split]["total"][i, 1, 0, :len(seq[2])] = torch.LongTensor(seq[2])
                start = self.max_input_len_s
                end = start + len(seq[3])
                self.sequences[split]["total"][i, 1, 0, start:end] = torch.LongTensor(seq[3])
                                              
                self.sequences[split]["total"][i, 1, 1, :len(seq[6])] = torch.LongTensor(seq[6])
                start = self.max_input_len_s 
                end = start + len(seq[7])
                self.sequences[split]["total"][i, 1, 1, start:end] = torch.LongTensor(seq[7])


                self.sequences[split]["total"][i, 2, 0, :len(seq[8])] = torch.LongTensor(seq[8])
                self.sequences[split]["total"][i, 2, 1, :len(seq[9])] = torch.LongTensor(seq[9])
                self.sequences[split]["total"][i, 2, 2, :len(seq[10])] = torch.LongTensor(seq[10])
                                
    def sample_batch(self, split, bs, cat="total", idxs=None):
        offset = self.offsets[split][cat]
        
        batch = {}

        # Decided not to reduce computation on here because it's all parallel
        # anyway and we don't want to run out of memory in cases where we
        # don't see the longest version quickly enough

        if idxs:
            seqs = self.sequences[split][cat].index_select(
                0, torch.LongTensor(idxs).to(
                    self.sequences[split][cat].device))
        else:
            seqs = self.sequences[split][cat][offset:offset + bs]
        batch["sequences"] = seqs.to(cfg.device)
        batch["attention_mask"] = make_attention_mask(seqs)                
        batch["loss_mask"] = make_loss_mask(seqs, self.max_input_len_k, self.max_input_len_s)
        batch["key"] = (cat, offset, offset + bs)

        offset += seqs.size(0)

        self.offsets[split][cat] = offset

        if split == "train" and offset + bs > len(self.sequences[split][cat]):
            return batch, True
        elif offset >= len(self.sequences[split][cat]):
            return batch, True
        else:
            return batch, False

    def reset_offsets(self, splits=["train", "test", "dev"],
                      shuffle=True, keys=None):
        if isinstance(splits, str):
            splits = [splits]
        
        for split in splits:
            if keys is None:
                keys = ["total"]
            
            for key in keys:
                self.offsets[split][key] = 0

            if shuffle:
                self.shuffle_sequences(split, keys)

    def shuffle_sequences(self, split="train", keys=None):
        if keys is None:
            
            keys = self.data[split].keys()

        for key in keys:
            
            idxs = list(range(len(self.data[split][key])))

            random.shuffle(idxs)

            self.sequences[split][key] = \
                self.sequences[split][key].index_select(
                    0, torch.LongTensor(idxs))

            temp = [self.data[split][key][i] for i in idxs]
            self.data[split][key] = temp

            temp = [self.masks[split][key][i] for i in idxs]
            self.masks[split][key] = temp


def make_attention_mask(sequences):
    return (sequences != 0).float().to(cfg.device)


def make_loss_mask(sequences, k, s):
    mask = (sequences != 0).float()
    
    mask[:, 0, 0, :k] = 0
    mask[:, 0, 1, :k] = 0
    mask[:, 1, 0, :s] = 0
    mask[:, 1, 1, :s] = 0
        
            
    return mask[:, :, :, 1:].to(cfg.device)


def get_generation_sequences(data, split, text_encoder, test,
                             max_e1=250, max_e2=250):
    sequences = []
    count = 0

        
    for input1, output1, input2, output2, input3, output3, input4, output4, sentence2, context, sentence5 in tqdm(data[split]["total"]):
        i1, o1, i2, o2, i3, o3, i4, o4, s2, c, s5 = do_example(text_encoder, input1, output1, input2, output2, input3, output3, input4, output4, sentence2, context, sentence5)          
        final = compile_final_sequence(i1, o1, i2, o2, i3, o3, i4, o4, s2, c, s5, text_encoder)
        sequences.append(final)
      
    return sequences


def do_example(text_encoder, input1, output1, input2, output2, input3, output3, input4, output4, sentence2, context, sentence5):
    
    #print(input1)
    #print(text_encoder.encode(input1))
    #s = "effects are #"
    #print(text_encoder.encode(s))
    #exit()
    i1 = text_encoder.encode(input1)
    o1 = text_encoder.encode(output1)
    i2 = text_encoder.encode(input2)
    o2 = text_encoder.encode(output2)
    i3 = text_encoder.encode(input3)
    o3 = text_encoder.encode(output3)
    i4 = text_encoder.encode(input4)
    o4 = text_encoder.encode(output4)    
    s2 = text_encoder.encode(sentence2)  
    c = text_encoder.encode(context)
    s5 = text_encoder.encode(sentence5)
    return i1, i2, i3, i4, o1, o2, o3, o4, s2, c, s5


def compile_final_sequence(i1, i2, i3, i4, o1, o2, o3, o4, s2, c, s5, text_encoder):
    final = []
    
    final.append(i1)
    final.append(o1+ [text_encoder.encoder["<END>"]])

    final.append(i2)
    final.append(o2 + [text_encoder.encoder["<END>"]])

    final.append(i3)
    final.append(o3 + [text_encoder.encoder["<END>"]])

    final.append(i4)
    final.append(o4 + [text_encoder.encoder["<END>"]])
    
    final.append(s2)
    final.append(c) 
    final.append(s5)
    return final


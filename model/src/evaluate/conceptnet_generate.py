import time
import torch

import src.evaluate.generate as base_generate
import src.evaluate.sampler as sampling
import utils.utils as utils
import src.data.config as cfg


def make_generator(opt, *args):
    return ConceptNetGenerator(opt, *args)


class ConceptNetGenerator(base_generate.Generator):
    def __init__(self, opt, model, model_knowledge, data_loader):
        self.opt = opt

        self.model = model
        self.model_knowledge = model_knowledge
        self.data_loader = data_loader

        self.sampler = sampling.make_sampler(
            opt.eval.sample, opt, data_loader)

    def reset_sequences(self):
        return []

    def generate(self, split="dev"):
        print("Generating Sequences")

        # Set evaluation mode
        self.model.eval()
        self.model_knowledge.eval()


        # Reset evaluation set for dataset split
        self.data_loader.reset_offsets(splits=split, shuffle=False)

        start = time.time()
        count = 0
        sequences = None

        # Reset generated sequence buffer
        sequences = self.reset_sequences()

        # Initialize progress bar
        bar = utils.set_progress_bar(
            self.data_loader.total_size[split] / 2)

        reset = False

        with torch.no_grad():
            # Cycle through development set
            while not reset:

                start = len(sequences)
                # Generate a single batch
                reset = self.generate_batch(sequences, split, bs=1)

                end = len(sequences)

                if not reset:
                    bar.update(end - start)
                else:
                    print(end)

                count += 1

                if cfg.toy and count > 10:
                    break
                if (self.opt.eval.gs != "full" and (count > opt.eval.gs)):
                    break

        torch.cuda.synchronize()
        print("{} generations completed in: {} s".format(
            split, time.time() - start))

        # Compute scores for sequences (e.g., BLEU, ROUGE)
        # Computes scores that the generator is initialized with
        # Change define_scorers to add more scorers as possibilities
        # avg_scores, indiv_scores = self.compute_sequence_scores(
        #     sequences, split)
        avg_scores, indiv_scores = None, None

        return sequences, avg_scores, indiv_scores

    def generate_batch(self, sequences, split, verbose=False, bs=1):
        # Sample batch from data loader
        batch, reset = self.data_loader.sample_batch(
            split, bs=bs, cat="total")

        input_ = batch["sequences"]
        attention_mask = batch["attention_mask"]
        start_idx_k = self.data_loader.max_input_len_k 
        max_end_len_k = self.data_loader.max_output_len_k
        start_idx_s = self.data_loader.max_input_len_s
        max_end_len_s = self.data_loader.max_output_len_s
                        
        #.unsqueeze(-1)
        for i in range(2):
                print("Iteration....", i)
                #decode knowledge
                      
                if i==0:
                    XMB_knowledge = input_[:, 0, i, :start_idx_k]  
                    MMB_knowledge = attention_mask[:, 0, i, :start_idx_k]
                    sampling_result, XMB = self.sampler.generate_sequence(XMB_knowledge, MMB_knowledge, self.model_knowledge, self.data_loader, start_idx_k, max_end_len_k)
                    print("Knowledge", sampling_result)
                    
                    previous_XMB_knowledge = XMB_knowledge[XMB_knowledge.nonzero().detach()]

                else:
                    MMB_knowledge  = attention_mask[:, 0, i, :start_idx_k]

                    XMB_knowledge_ = torch.LongTensor(MMB_knowledge.size(0), MMB_knowledge.size(-1)).fill_(0).cuda()
                    XMB_knowledge_[:, :previous_XMB_knowledge.size(-1)] = previous_XMB_knowledge
                    XMB_knowledge_[:, previous_XMB_knowledge.size(-1):XMB_knowledge.size(-1)] = XMB_knowledge
                    print("Knowledge_Input", XMB_knowledge_)
                    sampling_result, XMB = self.sampler.generate_sequence(XMB_knowledge_, MMB_knowledge, self.model_knowledge, self.data_loader, start_idx_k, max_end_len_k)
                    previous_XMB_knowledge = XMB_knowledge_[XMB_knowledge_.nonzero().detach()]
                       
                print("Knowledge", sampling_result)
                # Decode Sentence                
                MMB = attention_mask[:, 1, i, :start_idx_s]
                XMB_sentence = torch.LongTensor(MMB.size(0), MMB.size(-1)).fill_(0).cuda()
                XMB_sentence[:, :XMB.size(-1)]= XMB
                print("Sentence Input", XMB_sentence)
                sampling_result, XMB_knowledge = self.sampler.generate_sequence(XMB_sentence, MMB, self.model, self.data_loader, start_idx_s, max_end_len_s)

                print("Sentence", sampling_result)
                sequences.append(sampling_result)
        
        return reset

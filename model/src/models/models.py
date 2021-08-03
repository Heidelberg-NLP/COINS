from src.models.gpt2 import (GPT2DoubleHeadsModel, GPT2LMHeadModel)
#from src.models.gpt1 import (OpenAIGPTDoubleHeadsModel)
from .configuration_gpt2 import GPT2Config 
#from src.models.gpt import (LMModel, load_openai_pretrained_model)
import torch.nn as nn


def make_model(opt, n_vocab, n_ctx, n_special, load=True,
               return_acts=True, return_probs=False,
               clf_token="<CLASS>", answer_size=None):
    print(n_ctx)
    #self.output_hidden_states = config.output_hidden_states
    #self.output_attentions = config.output_attentions
    if opt.exp == "generation":
        #model = GPT2Model.from_pretrained(opt.net)
        model = GPT2LMHeadModel.from_pretrained('gpt2')
        #model = OpenAIGPTDoubleHeadsModel.from_pretrained('openai-gpt')
        #model(opt.net)
            #opt.net)#, True, True)#return_acts=return_acts,	
            #return_probs=return_probs)
    #elif opt.exp == "classification":
    #    model = ClfModel(
    #        opt.net, n_vocab, n_ctx, clf_token, answer_size)
    DEFAULT_CONFIG = GPT2Config 
    #if load:
    #    print("LOADING PRETRAINED TRANSFORMER")
    #    load_openai_pretrained_model(
    #        model.transformer, n_ctx=n_ctx, n_special=n_special)
    return model


def multi_gpu(model, devices):
    return nn.DataParallel(model)#, device_ids=devices)


def load_state_dict(model, state_dict):
    try:
        model.load_state_dict(state_dict)
    except RuntimeError:
        new_state_dict = {i[len("module."):]: j for i, j in state_dict.items()}
        model.load_state_dict(new_state_dict)

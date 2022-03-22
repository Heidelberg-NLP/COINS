This code is highly motivated from original COMET: https://github.com/atcbosselut/comet-commonsense.

Run the following script to load the data: 
```
python scripts/data/make_conceptnet_data_loader.py
```

Run the following script to train the COINS model:
```
bash run_train.sh
```
For inference run the following script:
```
bash run_test.sh
```
Checkout script/generate/generate_conceptnet_beam.py
```

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
                
                
```

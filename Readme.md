![](https://img.shields.io/github/last-commit/Heidelberg-NLP/COINS?color=blue) 

This directory contains the following parts of the 'COINS: Dynamically Generating COntextualized Inference Rules for Narrative Story Completion' experiment. 


## Where can you find the data? 
Please find the NSC data splits in ```data/``` folder

# Citation 

If you make use of the contents of this repository, please cite [the following paper](https://arxiv.org/pdf/2106.02497.pdf):

```bib
@inproceedings{paul-frank-2021-coins,
    title = "COINS: Dynamically Generating COntextualized Inference Rules for Narrative Story Completion",
    author = "Paul, Debjit  and Frank, Anette",
    booktitle = Proceedings of the Joint Conference of the 59th Annual Meeting of the Association for Computational Linguistics and the 11th International Joint Conference on Natural Language Processing (ACL-IJCNLP 2021),
    month = August,
    year = 2021,
    publisher = "Association for Computational Linguistics"
}
```

## Result
| Model | BLEU-1 | 
| :---: | :---: | 
| GPT-2 (small) | 16.66 |  
| T5 (small) |  20.67 |
| COINS (GPT-2 small) |  22.82 | 


## Setting Up the Environment
1. Create the `coins` environment using Anaconda

    ```
    conda create -n coins python=3.6
    ```

2. Activate the environment

  ```
  source activate coins
  ```

3. Install the requirements in the environment:

```
pip install -r requirements.txt
```

Install pytorch that supports cuda8 cuda 8:
```
pip install torch==0.4.1
```
```
## Requirements 
~~~~
python3.8+
pip3 install torch torchvision torchaudio
~~~~

Install the library and dependencies
~~~~
git clone https://github.com/huggingface/transformers
cd transformers
pip install .
pip install -r ./examples/requirements.txt
pip install tensorflow
pip install ftfy==5.1
conda install -c conda-forge spacy
python -m spacy download en
pip install tensorboardX
pip install tqdm
pip install pandas
pip install ipython
~~~~



# Any Issue?
For any questions or issues about this repository, please write to paul@cl.uni-heidelberg.de

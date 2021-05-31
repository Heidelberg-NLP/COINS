![](https://img.shields.io/github/last-commit/Heidelberg-NLP/COINS?color=blue) 


This directory contains the following parts of the 'COINS: Dynamically Generating COntextualized Inference Rules for Narrative Story Completion' experiment. 

<p align="center">
  <img src="" alt="COINS">
</p>

## Reference

If you make use of the contents of this repository, please cite [the following paper](https://www.aclweb.org/anthology/N19-1368):

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
## Requirements 
~~~~
- python3.8+
- pytorch
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

## Recreating Results
<p align="center">
  <img src="" alt="Results">
</p>


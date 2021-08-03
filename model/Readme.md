This code is highly motivated from original COMET: https://github.com/atcbosselut/comet-commonsense.
Run:
```
python scripts/data/make_conceptnet_data_loader.py
python src/main.py --experiment_type atomic --experiment_num #
python scripts/evaluate/evaluate_atomic_generation_model.py --split $DATASET_SPLIT --model_name /path/to/model/file
```

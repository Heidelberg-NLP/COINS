# create a file with tasks to execute
python_exec=/path/to/the/python/environment
# CHANGEME - your script for running one single file.
#JOB_SCRIPT="PYTHONPATH=. $python_exec src/main.py --experiment_type conceptnet --experiment_num 0"
$python_exec scripts/generate/generate_conceptnet_beam_search.py --beam 1 --path prev_sentence --seed 42 --split dev --model_name models/story.pickle --model_knowledge_name models/knowledge.pickle

# create a file with tasks to execute
DATA_BASE_DIR='./'
#ls ${DATA_BASE_DIR}/*.chunk.* > ${DATA_BASE_DIR}/chunk_files.txt

job_name_base=generate_

#while read input_file; do
#echo "$input_file"
JOB_NAME=${job_name_base}_${input_file##*/}
  
job_mem=10G
job_time=48:00:00   # HH:mm:ss
job_cpus_per_task=8
gpu_selector='gpu:mem11g:1'
# params below not changed very often
# CHANGEME - path to your environment python (when environment is activated write "which python")
python_exec=/home/mitarb/paul/anaconda3/envs/dual/bin/python
JOB_NAME=${JOB_NAME}_$(date +%y-%m-%d-%H-%M-%S)
# CHANGEME - your script for running one single file.
#JOB_SCRIPT="PYTHONPATH=. $python_exec src/main.py --experiment_type conceptnet --experiment_num 0"
JOB_SCRIPT="PYTHONPATH=. $python_exec scripts/generate/generate_conceptnet_beam_search.py --beam 1 --path prev_sentence --seed 42 --split dev --model_name models/story_forward_backward.pickle --model_knowledge_name models/knowledge_forward_backward.pickle"
#DO NOT change anything here!
echo "bash ~/cluster_cmd/run_sbatch.sh \"${JOB_SCRIPT}\" ${JOB_NAME} ${job_mem} ${job_time} ${job_cpus_per_task} ${gpu_selector}"
bash ~/cluster_cmd/run_sbatch_gpulong.sh "${JOB_SCRIPT}" ${JOB_NAME} ${job_mem} ${job_time} ${job_cpus_per_task} ${gpu_selector}


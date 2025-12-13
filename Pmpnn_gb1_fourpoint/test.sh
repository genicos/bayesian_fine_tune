#!/bin/bash
#SBATCH --cluster=whale
#SBATCH --partition=long
#SBATCH --account=researchers
#SBATCH --job-name=gb1_4pt_FT
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=50G
#SBATCH --output=/home/nico/logs/%j.out
#SBATCH --error=/home/nico/logs/%j.err



gpu_id=$1
job_file="experiments/gen_jobs/exp_4/jobs/jobs_${gpu_id}.txt"

cat $job_file | parallel -j 2 {}
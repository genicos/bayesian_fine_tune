import random

learning_rates = [0.01, 0.1, 1, 10]
loss_types = ["bayesian", "reward_weighted_SFT"]

alphas = [0.01,0.1, 1, 10, 100]
Bs = [4, 8]

random_seeds = [0, 1, 2, 3]

job_id = 0

jobs = []

for B in Bs:
    for learning_rate in learning_rates:
        for alpha in alphas:
            for random_seed in random_seeds:
                #jobs.append(f"python3 main.py --loss_type reward_weighted_SFT --alpha {alpha} --B {B} --learning_rate {learning_rate:.4g} --job_id {job_id}  --random_seed {random_seed} --save_results")
                #job_id += 1
                jobs.append(f"python3 wt_1ball_real.py --loss_type bayesian --alpha {alpha} --B {B} --learning_rate {learning_rate:.4g} --job_id {job_id} --random_seed {random_seed} --save_results")
                job_id += 1

#Randomly shuffle jobs and split into GPUs
num_gpus = 3
random.shuffle(jobs)

for gpu_id in range(num_gpus):
    job_subset = jobs[gpu_id::num_gpus]

    with open(f"jobs/jobs_{gpu_id}.txt", "w") as f:
        for job in job_subset:
            f.write(job + f" --gpu_id {gpu_id}\n")

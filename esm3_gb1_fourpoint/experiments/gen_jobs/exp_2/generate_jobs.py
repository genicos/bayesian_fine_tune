import random

learning_rates = [10**-4,10**-3.5, 10**-3, 10**-2.5, 10**-2, 10**-1.5, 10**-1, 10**-0.5, 1]
loss_types = ["bayesian", "reward_weighted_SFT"]

alphas = [0.1, 0.316, 1, 3.16, 10]
Bs = [4, 8]

job_id = 0

jobs = []

for B in Bs:
    for learning_rate in learning_rates:
        for alpha in alphas:
            jobs.append(f"python3 main.py --loss_type reward_weighted_SFT --alpha {alpha} --B {B} --learning_rate {learning_rate:.4g} --job_id {job_id}  --random_seed 0 --save_results")
            job_id += 1
            jobs.append(f"python3 main.py --loss_type bayesian --alpha {alpha} --B {B} --learning_rate {learning_rate:.4g} --job_id {job_id} --random_seed 0 --save_results")
            job_id += 1

#Randomly shuffle jobs and split into GPUs
num_gpus = 2
random.shuffle(jobs)

for gpu_id in range(num_gpus):
    job_subset = jobs[gpu_id::num_gpus]

    with open(f"jobs/jobs_{gpu_id}.txt", "w") as f:
        for job in job_subset:
            f.write(job + f" --gpu_id {gpu_id}\n")

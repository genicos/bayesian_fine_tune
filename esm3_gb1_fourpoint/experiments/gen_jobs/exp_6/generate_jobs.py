import random

learning_rates = [10**-4, 10**-3,  10**-2, 10**-1, 1]
loss_types = ["bayesian", "reward_weighted_SFT"]

alphas = [0.5, 1]
Bs = [4, 8]
random_seeds = [0, 1, 2, 3]

job_id = 0

jobs = []

for B in Bs:
    for learning_rate in learning_rates:
        for alpha in alphas:
            for random_seed in random_seeds:
                jobs.append(f"python3 wt_1ball_bool.py --loss_type reward_weighted_SFT --alpha {alpha} --B {B} --learning_rate {learning_rate:.4g} --job_id {job_id}  --random_seed {random_seed} --save_results --result_file experiments/results/exp_6/results_{job_id}.pkl")
                job_id += 1
                jobs.append(f"python3 wt_1ball_bool.py --loss_type bayesian --alpha {alpha} --B {B} --learning_rate {learning_rate:.4g} --job_id {job_id} --random_seed {random_seed} --save_results --result_file experiments/results/exp_6/results_{job_id}.pkl")
                job_id += 1
                jobs.append(f"python3 wt_1ball_bool.py --loss_type bayesian --rescale_loss --alpha {alpha} --B {B} --learning_rate {learning_rate:.4g} --job_id {job_id} --random_seed {random_seed} --save_results --result_file experiments/results/exp_6/results_{job_id}.pkl")
                job_id += 1


#HERE

#Group jobs by random_seed
jobs_by_seed = {}
for job in jobs:
    for seed in random_seeds:
        if f"--random_seed {seed}" in job:
            if seed not in jobs_by_seed:
                jobs_by_seed[seed] = []
            jobs_by_seed[seed].append(job)
            break

#Shuffle within each seed group
for seed in jobs_by_seed:
    random.shuffle(jobs_by_seed[seed])

#Distribute jobs across GPUs maintaining seed order
num_gpus = 4
gpu_jobs = [[] for _ in range(num_gpus)]

for seed in sorted(jobs_by_seed.keys()):
    seed_jobs = jobs_by_seed[seed]
    for i, job in enumerate(seed_jobs):
        gpu_id = i % num_gpus
        gpu_jobs[gpu_id].append((seed, job))

#Reorganize each GPU's jobs: all seed 0 first, then seed 1, etc.
for gpu_id in range(num_gpus):
    gpu_jobs[gpu_id].sort(key=lambda x: x[0])
    gpu_jobs[gpu_id] = [job for _, job in gpu_jobs[gpu_id]]

for gpu_id in range(num_gpus):
    with open(f"jobs/jobs_{gpu_id}.txt", "w") as f:
        for job in gpu_jobs[gpu_id]:
            f.write(job + f" --gpu_id {gpu_id}\n")

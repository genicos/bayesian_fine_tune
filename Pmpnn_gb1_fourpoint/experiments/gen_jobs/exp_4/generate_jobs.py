import random

learning_rates_bayesian = [10**-2.8,10**-2.6, 10**-2.4, 10**-2.2]
alphas_bayesian = [10**-1.2,10**-1.1, 10**-1, 10**-0.9, 10**-0.8]


learning_rates_reward_weighted_SFT = [10**-4.4, 10**-4.2, 10**-4, 10**-3.8]
alphas_reward_weighted_SFT = [10**-1.4, 10**-1.2, 10**-1, 10**-0.8]

random_seeds = [4, 5, 6, 7]

job_id = 0

jobs = []


for learning_rate_bayesian in learning_rates_bayesian:
    for alpha_bayesian in alphas_bayesian:
        for random_seed in random_seeds:
            jobs.append(f"python3 main.py --training_set wt_2ball_low_pass --loss_type bayesian --alpha {alpha_bayesian:.4g} --B 1 --learning_rate {learning_rate_bayesian:.4g} --job_id {job_id} --random_seed {random_seed} --save_results --result_file experiments/results/exp_4/results_{job_id}.pkl")
            job_id += 1
for learning_rate_reward_weighted_SFT in learning_rates_reward_weighted_SFT:
    for alpha_reward_weighted_SFT in alphas_reward_weighted_SFT:
        for random_seed in random_seeds:
            jobs.append(f"python3 main.py --training_set wt_2ball_low_pass --loss_type reward_weighted_SFT --alpha {alpha_reward_weighted_SFT:.4g} --B 1 --learning_rate {learning_rate_reward_weighted_SFT:.4g} --job_id {job_id}  --random_seed {random_seed} --save_results --result_file experiments/results/exp_4/results_{job_id}.pkl")
            job_id += 1



#Randomly shuffle jobs and split into GPUs
num_gpus = 2
random.shuffle(jobs)

for gpu_id in range(num_gpus):
    job_subset = jobs[gpu_id::num_gpus]

    with open(f"jobs/jobs_{gpu_id}.txt", "w") as f:
        for job in job_subset:
            f.write(job + f" --gpu_id {gpu_id}\n")

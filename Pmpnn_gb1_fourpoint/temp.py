import argparse
from operator import ge
import os
import pickle

parser = argparse.ArgumentParser(description="wt_1ball_real")

parser.add_argument("--loss_type", type=str, required=False, default="bayesian")
parser.add_argument("--alpha", type=float, required=False, default=0.1585)
parser.add_argument("--B", type=int, required=False, default=1)
parser.add_argument("--epochs", type=int, required=False, default=100)
parser.add_argument("--learning_rate", type=float, required=False, default=0.001)
parser.add_argument("--batch_size", type=int, required=False, default=16) 
parser.add_argument("--train_proportion", type=float, required=False, default=0.8)
parser.add_argument("--patience", type=int, required=False, default=3)
parser.add_argument("--eval_N", type=int, required=False, default=1000)
parser.add_argument("--gpu_id", type=int, required=False, default=0)
parser.add_argument("--r", type=int, required=False, default=4)
parser.add_argument("--lora_alpha", type=int, required=False, default=2)
parser.add_argument("--random_seed", type=int, required=False, default=0)
parser.add_argument("--job_id", type=int, required=False, default=-1)
parser.add_argument("--rescale_loss", action="store_true")
parser.add_argument("--save_results", action="store_true")
parser.add_argument("--result_file", type=str, required=False, default="")


parser.add_argument("--boolean", action="store_true")
parser.add_argument("--training_set", type=str, required=False, default="wt_2ball",
    choices=["wt_1ball", "wt_2ball", "wt_2ball_low_pass","AAAA_2ball", "full_set_low_pass"])


args = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)


if args.save_results and ( (args.job_id != -1 and os.path.exists(f"results/results_{args.job_id}.pkl")) or (args.result_file != "" and os.path.exists(args.result_file))):
    print("Results already exist, skipping")
    exit()


from fine_tuning import *
from evaluate_model import *
import json


# Set seeds for reproducibility/determinism
torch.manual_seed(args.random_seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(args.random_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

lora_cfg = LoraConfig(
    r=args.r,
    lora_alpha=args.lora_alpha,
    lora_dropout=0.0,
    bias="none",
    target_modules=[
        "W1",
        "W2",
        "W3",
        "W11",
        "W12",
        "W13",
    ]
)
model = get_peft_model(base_model, lora_cfg)




if args.training_set == "wt_1ball":
    print("Training on wt_1ball")
    subset = hamming_1_ball()
elif args.training_set == "wt_2ball":
    print("Training on wt_2ball")
    subset = hamming_2_ball()
elif args.training_set == "wt_2ball_low_pass":
    print("Training on wt_2ball_low_pass")
    subset = hamming_2_ball_low_pass()
elif args.training_set == "full_set_low_pass":
    print("Training on full_set_low_pass")
    subset = full_set_low_pass()
elif args.training_set == "AAAA_2ball":
    print("Training on AAAA_2ball")
    subset = hamming_2_ball([1,1,1,1])

print("Subset statistics:")
print("Length of subset: ", len(subset))
average_assay_value = sum(assay_values[subset]) / len(subset)
print("Average assay value: ", average_assay_value)
num_assays_above_wildtype = sum(1 for i in subset if assay_values[i] >= 1)
print("Number of assays above wildtype: ", num_assays_above_wildtype)
if args.boolean:
    num_assays_above_threshold = sum(1 for i in subset if assay_values[i] >= args.alpha)
    print("Number of assays above threshold: ", num_assays_above_threshold)
#exit()


if False:
    print("BASE_MODEL")
    generated_samples = generate_N_mutations(model, temp=1, N=1000)
    results = evaluate_samples(subset, generated_samples)
    assays_above_threshold = []
    for code in generated_samples:
        index = mutations_to_index[code[0], code[1], code[2], code[3]]
        if index != -1 and assay_values[index] >= args.alpha:
            assays_above_threshold.append(assay_values[index])

    print(f"Number of assays above threshold: {len(assays_above_threshold)}")
    print(f"Average assay value above threshold: {sum(assays_above_threshold) / len(assays_above_threshold)}")
    print("--------------------------------")


if args.loss_type == "bayesian" or args.loss_type == "reward_weighted_SFT":
    train_loss_history, validation_loss_history = weighted_fine_tuning_train(model, assay_indices=subset, alpha=args.alpha, B=args.B, epochs=args.epochs, learning_rate=args.learning_rate, batch_size=args.batch_size, train_proportion=args.train_proportion, random_seed=args.random_seed, loss_type=args.loss_type, patience=args.patience, rescale_loss=args.rescale_loss, threshold=args.boolean)
    


generated_samples = generate_N_mutations(model, temp=1, N=1000)
results = evaluate_samples(subset, generated_samples)

sample_assay_values = []
assays_above_threshold = []
assays_above_threshold_not_in_subset = []

for code in generated_samples:
    index = mutations_to_index[code[0], code[1], code[2], code[3]]
    if index != -1:
        sample_assay_values.append(assay_values[index])
        if assay_values[index] >= args.alpha:
            assays_above_threshold.append(assay_values[index])
        if index not in subset and assay_values[index] >= args.alpha:
            assays_above_threshold_not_in_subset.append(assay_values[index])

print(f"Number of sample assays: {len(sample_assay_values)}")
print(f"Number of assays above threshold: {len(assays_above_threshold)}")
print(f"Number of assays above threshold not in subset: {len(assays_above_threshold_not_in_subset)}")
print(f"Proportion of assays above threshold not in subset: {len(assays_above_threshold_not_in_subset) / (results['num_assayed'] - results['num_in_train'])}")

results["sample_assay_values"] = torch.tensor(sample_assay_values)
results["assays_above_threshold"] = torch.tensor(assays_above_threshold)
results["assays_above_threshold_not_in_subset"] = torch.tensor(assays_above_threshold_not_in_subset)


if args.save_results:
    save_object = {
        "results": results,
        "args": args.__dict__
    }
    if args.result_file != "":
        result_file = args.result_file
    else:
        result_file = f"results/results_{args.job_id}.pkl"
    
    with open(result_file, "wb") as f:
        pickle.dump(save_object, f)


import argparse
import os
import pickle

parser = argparse.ArgumentParser(description="wt_1ball_real")

parser.add_argument("--loss_type", type=str, required=False, default="bayesian")
parser.add_argument("--alpha", type=float, required=False, default=1)
parser.add_argument("--B", type=int, required=False, default=16)
parser.add_argument("--epochs", type=int, required=False, default=100)
parser.add_argument("--learning_rate", type=float, required=False, default=0.00001)
parser.add_argument("--batch_size", type=int, required=False, default=16)
parser.add_argument("--train_proportion", type=float, required=False, default=0.8)
parser.add_argument("--patience", type=int, required=False, default=3)
parser.add_argument("--eval_N", type=int, required=False, default=1000)
parser.add_argument("--gpu_id", type=int, required=False, default=0)
parser.add_argument("--r", type=int, required=False, default=8)
parser.add_argument("--lora_alpha", type=int, required=False, default=16)
parser.add_argument("--random_seed", type=int, required=False, default=0)
parser.add_argument("--job_id", type=int, required=False, default=-1)
parser.add_argument("--rescale_loss", action="store_true")
parser.add_argument("--save_results", action="store_true")

args = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)


if args.job_id != -1 and os.path.exists(f"results/results_{args.job_id}.pkl"):
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




wt_1_ball = hamming_1_ball()
subset = wt_1_ball


print("BASE_MODEL")
generated_samples = generate_N_mutations(model, temp=1, N=1000)
results = evaluate_samples(subset, generated_samples)


if args.loss_type == "bayesian" or args.loss_type == "reward_weighted_SFT":
    train_loss_history, validation_loss_history = weighted_fine_tuning_train(model, assay_indices=subset, alpha=args.alpha, B=args.B, epochs=args.epochs, learning_rate=args.learning_rate, batch_size=args.batch_size, train_proportion=args.train_proportion, random_seed=args.random_seed, loss_type=args.loss_type, patience=args.patience, rescale_loss=args.rescale_loss)
    


generated_samples = generate_N_mutations(model, temp=1, N=1000)
results = evaluate_samples(subset, generated_samples)

if args.save_results:
    save_object = {
        "results": results,
        "args": args.__dict__
    }
    with open(f"results/results_{args.job_id}.pkl", "wb") as f:
        pickle.dump(save_object, f)


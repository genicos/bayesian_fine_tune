import pickle
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import sys
import os

take_0_dir = Path(__file__).parent.parent.parent.parent
original_cwd = os.getcwd()
os.chdir(take_0_dir)

sys.path.insert(0, str(take_0_dir))
from mutation_sets import hamming_1_ball
from prepare_gb1_4_point_data import mutations_to_index, assay_values

os.chdir(original_cwd)

results_dir = Path("../../results/exp_1")
output_dir = Path(".")

alpha = 1.0
Bs = [4, 8]

train_indices_set = set(x.item() for x in hamming_1_ball())
print(train_indices_set)
print(len(train_indices_set))
print("AAA")

metrics = ["adjusted_average_assay_value", "best_assay_value", "num_in_train", "unique_indecies_sampled", "average_assay_value_not_in_train", "best_assay_value_not_in_training_set"]

def load_all_results():
    all_results = {}
    learning_rates_set = set()
    loss_types_set = set()
    
    for pkl_file in sorted(results_dir.glob("results_*.pkl")):
        try:
            with open(pkl_file, 'rb') as f:
                data = pickle.load(f)
                args = data['args']
                results = data['results']
                
                loss_type = args.get('loss_type', 'bayesian')
                file_alpha = args.get('alpha', 1.0)
                B = args.get('B', 4)
                learning_rate = args.get('learning_rate', 1.0)
                
                if loss_type == "reward_weighted_SFT":
                    file_alpha = None
                elif file_alpha != alpha:
                    continue
                
                learning_rates_set.add(learning_rate)
                loss_types_set.add(loss_type)
                
                key = (loss_type, file_alpha, B, learning_rate)
                all_results[key] = results
        except Exception as e:
            print(f"Error loading {pkl_file}: {e}")
            continue
    
    learning_rates = sorted(list(learning_rates_set))
    loss_types = sorted(list(loss_types_set))
    
    return all_results, learning_rates, loss_types

def calculate_average_assay_value_not_in_train(results):
    if not isinstance(results, dict) or 'sample_codes' not in results:
        return np.nan
    
    sample_codes = results['sample_codes']
    assay_values_not_in_train = []

    count = 0
    
    for code in sample_codes:
        index = mutations_to_index[code[0], code[1], code[2], code[3]].item()
        if index != -1:
            if index not in train_indices_set:
                assay_values_not_in_train.append(assay_values[index].item())
            else:
                count += 1
        else:
            assay_values_not_in_train.append(0)

    print(f"Number of samples in training set1: {count}")
    print(f"Number of samples in training set2: {results['num_in_train']}")
    if count != results['num_in_train']:
        print(f"Number of samples in training set1 and training set2 do not match")
        exit()
    
    if len(assay_values_not_in_train) == 0:
        return np.nan
    
    return np.mean(assay_values_not_in_train)

def calculate_best_assay_value_not_in_train(results):
    if not isinstance(results, dict) or 'sample_codes' not in results:
        return np.nan
    
    sample_codes = results['sample_codes']
    assay_values_not_in_train = []
    
    for code in sample_codes:
        index = mutations_to_index[code[0], code[1], code[2], code[3]].item()
        if index != -1:
            if index not in train_indices_set:
                assay_values_not_in_train.append(assay_values[index].item())
    
    if len(assay_values_not_in_train) == 0:
        return np.nan
    
    return np.max(assay_values_not_in_train)

def get_metric_value(results, metric_name):
    if isinstance(results, dict):
        if metric_name == "average_assay_value_not_in_train":
            return calculate_average_assay_value_not_in_train(results)
        elif metric_name == "best_assay_value_not_in_training_set":
            return calculate_best_assay_value_not_in_train(results)
        return results.get(metric_name, np.nan)
    return np.nan

def create_heatmap_figure(all_results, metric_name, learning_rates, loss_types):
    n_cols = 1
    n_rows = len(loss_types)
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6, n_rows * 3))
    if n_rows == 1:
        axes = [axes]
    else:
        axes = axes.flatten()
    
    all_values = []
    
    for loss_type in loss_types:
        if loss_type == "reward_weighted_SFT":
            for b_idx, B in enumerate(Bs):
                for lr in learning_rates:
                    key = (loss_type, None, B, lr)
                    if key in all_results:
                        value = get_metric_value(all_results[key], metric_name)
                        if not np.isnan(value):
                            all_values.append(value)
        else:
            for b_idx, B in enumerate(Bs):
                for lr in learning_rates:
                    key = (loss_type, alpha, B, lr)
                    if key in all_results:
                        value = get_metric_value(all_results[key], metric_name)
                        if not np.isnan(value):
                            all_values.append(value)
    
    if len(all_values) > 0:
        vmin = min(all_values)
        vmax = max(all_values)
    else:
        vmin = 0
        vmax = 1
    
    for row_idx, loss_type in enumerate(loss_types):
        ax = axes[row_idx]
        
        heatmap_data = np.full((len(Bs), len(learning_rates)), np.nan)
        for b_idx, B in enumerate(Bs):
            for lr_idx, lr in enumerate(learning_rates):
                if loss_type == "reward_weighted_SFT":
                    key = (loss_type, None, B, lr)
                else:
                    key = (loss_type, alpha, B, lr)
                if key in all_results:
                    value = get_metric_value(all_results[key], metric_name)
                    heatmap_data[b_idx, lr_idx] = value
        
        fmt_str = '.0f' if metric_name in ['num_in_train', 'unique_indecies_sampled'] else '.3f'
        sns.heatmap(heatmap_data, ax=ax, cmap='viridis', annot=True, fmt=fmt_str,
                   xticklabels=[f"LR={lr:.2e}" for lr in learning_rates],
                   yticklabels=[f"B={B}" for B in Bs],
                   vmin=vmin, vmax=vmax, cbar_kws={'label': metric_name})
        
        ax.set_title(loss_type, fontsize=12)
        ax.set_xlabel("Learning Rate", fontsize=10)
        ax.set_ylabel("B", fontsize=10)
    
    plt.suptitle(f"{metric_name} (α={alpha})", fontsize=14, y=1.02)
    plt.tight_layout()
    
    output_path = output_dir / f"{metric_name}.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")

def main():
    print("Loading all results...")
    all_results, learning_rates, loss_types = load_all_results()
    print(f"Loaded {len(all_results)} result files")
    print(f"Found learning rates: {learning_rates}")
    print(f"Found loss types: {loss_types}")
    
    for metric_name in metrics:
        print(f"Creating heatmap for {metric_name}...")
        create_heatmap_figure(all_results, metric_name, learning_rates, loss_types)
    
    print("All visualizations created!")

if __name__ == "__main__":
    main()


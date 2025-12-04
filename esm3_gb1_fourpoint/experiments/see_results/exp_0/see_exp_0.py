import pickle
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

results_dir = Path("../results")
output_dir = Path(".")

learning_rates = [1e-4, 1e-3, 1e-2, 1e-1, 1]
alphas = [0.1, 0.3, 1, 3, 10]
Bs = [1, 2, 4]
rs = [6, 8, 10]
lora_alphas = [8, 16]

metrics = ["adjusted_average_assay_value", "best_assay_value", "num_in_train", "unique_indecies_sampled"]

def load_all_results():
    all_results = {}
    
    for pkl_file in sorted(results_dir.glob("results_*.pkl")):
        try:
            with open(pkl_file, 'rb') as f:
                data = pickle.load(f)
                args = data['args']
                results = data['results']
                
                loss_type = args.get('loss_type', 'bayesian')
                alpha = args.get('alpha', 1.0)
                B = args.get('B', 4)
                learning_rate = args.get('learning_rate', 1.0)
                r = args.get('r', 8)
                lora_alpha = args.get('lora_alpha', 16)
                
                if loss_type == "reward_weighted_SFT":
                    alpha = None
                
                key = (loss_type, alpha, B, learning_rate, r, lora_alpha)
                all_results[key] = results
        except Exception as e:
            print(f"Error loading {pkl_file}: {e}")
            continue
    
    return all_results

def get_x_labels():
    x_labels = ["reward_weighted_SFT"]
    x_labels.extend([f"α={alpha}" for alpha in alphas])
    return x_labels

def get_metric_value(results, metric_name):
    if isinstance(results, dict):
        return results.get(metric_name, np.nan)
    return np.nan

def create_heatmap_figure(all_results, metric_name, lora_alpha):
    x_labels = get_x_labels()
    y_labels = [f"B={B}" for B in Bs]
    
    n_rows = len(rs)
    n_cols = len(learning_rates)
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 4, n_rows * 3))
    if n_rows == 1:
        axes = axes.reshape(1, -1)
    if n_cols == 1:
        axes = axes.reshape(-1, 1)
    
    all_values = []
    
    for row_idx, r in enumerate(rs):
        for col_idx, lr in enumerate(learning_rates):
            for b_idx, B in enumerate(Bs):
                for x_idx, x_label in enumerate(x_labels):
                    if x_idx == 0:
                        loss_type = "reward_weighted_SFT"
                        alpha = None
                    else:
                        loss_type = "bayesian"
                        alpha = alphas[x_idx - 1]
                    
                    key = (loss_type, alpha, B, lr, r, lora_alpha)
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
    
    for row_idx, r in enumerate(rs):
        for col_idx, lr in enumerate(learning_rates):
            ax = axes[row_idx, col_idx]
            
            heatmap_data = np.full((len(Bs), len(x_labels)), np.nan)
            
            for b_idx, B in enumerate(Bs):
                for x_idx, x_label in enumerate(x_labels):
                    if x_idx == 0:
                        loss_type = "reward_weighted_SFT"
                        alpha = None
                    else:
                        loss_type = "bayesian"
                        alpha = alphas[x_idx - 1]
                    
                    key = (loss_type, alpha, B, lr, r, lora_alpha)
                    if key in all_results:
                        value = get_metric_value(all_results[key], metric_name)
                        heatmap_data[b_idx, x_idx] = value
            
            fmt_str = '.0f' if metric_name == 'num_in_train' or metric_name == 'unique_indecies_sampled' else '.3f'
            sns.heatmap(heatmap_data, ax=ax, cmap='viridis', annot=True, fmt=fmt_str,
                       xticklabels=x_labels, yticklabels=y_labels,
                       vmin=vmin, vmax=vmax, cbar_kws={'label': metric_name})
            
            if row_idx == 0:
                ax.set_title(f"LR={lr:.0e}", fontsize=10)
            if col_idx == 0:
                ax.set_ylabel(f"r={r}", fontsize=10)
    
    plt.suptitle(f"{metric_name} (lora_alpha={lora_alpha})", fontsize=14, y=1.02)
    plt.tight_layout()
    
    output_path = output_dir / f"{metric_name}_lora_alpha_{lora_alpha}.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")

def main():
    print("Loading all results...")
    all_results = load_all_results()
    print(f"Loaded {len(all_results)} result files")
    
    for metric_name in metrics:
        for lora_alpha in lora_alphas:
            print(f"Creating heatmap for {metric_name} with lora_alpha={lora_alpha}...")
            create_heatmap_figure(all_results, metric_name, lora_alpha)
    
    print("All visualizations created!")

if __name__ == "__main__":
    main()


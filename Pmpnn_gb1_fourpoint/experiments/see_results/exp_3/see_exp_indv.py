import os
import pickle
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
import torch

base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
data_dir = os.path.join(base_dir, "data")

mutations_to_index = torch.load(os.path.join(data_dir, "mutations_to_index.pt"), map_location="cpu")
assay_values = torch.load(os.path.join(data_dir, "assay_values.pt"), map_location="cpu")


def calculate_assays_above_threshold(sample_codes, threshold):
    """
    Calculate the number of assays above threshold from sample codes.
    
    Args:
        sample_codes: list of mutation codes [code0, code1, code2, code3]
        threshold: alpha threshold value
        
    Returns:
        count: number of assays above threshold
    """
    count = 0
    for code in sample_codes:
        index = mutations_to_index[code[0], code[1], code[2], code[3]]
        if index != -1 and assay_values[index] >= threshold:
            count += 1
    return count


def load_results(results_dir):
    """
    Load all result pickle files from the results directory.

    Returns:
        entries: list of dicts with keys
            B, alpha, learning_rate, loss_type, rescale_loss, random_seed, and all metrics from the results dict.
    """
    entries = []

    if not os.path.isdir(results_dir):
        raise FileNotFoundError(f"Results directory not found: {results_dir}")

    for fname in os.listdir(results_dir):
        if not fname.endswith(".pkl"):
            continue
        fpath = os.path.join(results_dir, fname)
        with open(fpath, "rb") as f:
            obj = pickle.load(f)

        args = obj.get("args", {})
        res = obj.get("results", {})

        loss_type = args.get("loss_type", "bayesian")
        rescale_loss = args.get("rescale_loss", False)
        
        if rescale_loss and loss_type == "bayesian":
            loss_type = "bayesian_rescale"
        
        entry = {
            "B": args.get("B"),
            "alpha": args.get("alpha"),
            "learning_rate": args.get("learning_rate"),
            "loss_type": loss_type,
            "random_seed": args.get("random_seed"),
        }
        entry.update(res)
        
        sample_codes = res.get("sample_codes", [])
        if sample_codes:
            threshold = args.get("alpha", 1.0)
            entry["assays_above_threshold"] = calculate_assays_above_threshold(sample_codes, threshold)
        else:
            entry["assays_above_threshold"] = np.nan
        
        entries.append(entry)

    if not entries:
        raise RuntimeError(f"No result pickle files found in {results_dir}")

    return entries


def average_across_seeds(entries):
    """
    Group entries by (B, alpha, learning_rate, loss_type) and average metrics across random_seed.
    Also track num_runs_finished for each parameter set.

    Returns:
        averaged_entries: list of dicts with averaged metrics and num_runs_finished
    """
    groups = defaultdict(list)
    
    for e in entries:
        key = (e["B"], e["alpha"], e["learning_rate"], e["loss_type"])
        groups[key].append(e)
    
    averaged_entries = []
    metric_keys = [
        "num_in_train",
        "adjusted_average_assay_value_not_in_train",
        "best_assay_value_not_in_train",
        "unique_indecies_sampled",
        "num_assayed",
        "average_assay_value",
        "adjusted_average_assay_value",
        "best_assay_value",
        "assays_above_threshold",
    ]
    
    for key, group_entries in groups.items():
        B, alpha, lr, lt = key
        num_runs_finished = len(group_entries)
        
        averaged_entry = {
            "B": B,
            "alpha": alpha,
            "learning_rate": lr,
            "loss_type": lt,
            "num_runs_finished": num_runs_finished,
        }
        
        for metric in metric_keys:
            values = [e.get(metric) for e in group_entries if metric in e]
            if values:
                averaged_entry[metric] = np.mean(values)
            else:
                averaged_entry[metric] = np.nan
        
        averaged_entries.append(averaged_entry)
    
    return averaged_entries


def build_grids_with_seeds(entries, metric):
    """
    Build grids with sub-cells for each seed and average.
    Groups entries by (B, alpha, learning_rate, loss_type) and tracks individual seeds.

    Returns:
        loss_types: sorted list of unique loss_type values
        alphas: sorted list of unique alpha values
        learning_rates: sorted list of unique learning_rate values
        grids: dict mapping loss_type -> nested structure: (lr_idx, alpha_idx) -> list of (seed, value) tuples + average
        max_seeds: maximum number of seeds across all parameter combinations
    """
    alphas = sorted({e["alpha"] for e in entries})
    learning_rates = sorted({e["learning_rate"] for e in entries})
    loss_types = sorted({e["loss_type"] for e in entries})

    alpha_index = {a: i for i, a in enumerate(alphas)}
    lr_index = {lr: i for i, lr in enumerate(learning_rates)}

    grids = {}
    for lt in loss_types:
        grids[lt] = {}

    groups = defaultdict(list)
    for e in entries:
        key = (e["alpha"], e["learning_rate"], e["loss_type"])
        groups[key].append(e)

    max_seeds = 0
    for key, group_entries in groups.items():
        a, lr, lt = key
        if lt not in grids:
            continue
        i_lr = lr_index[lr]
        i_a = alpha_index[a]
        
        seed_values = []
        for e in group_entries:
            seed = e.get("random_seed")
            value = e.get(metric, np.nan)
            if not np.isnan(value):
                seed_values.append((seed, value))
        
        if seed_values:
            values = [v for _, v in seed_values]
            avg_value = np.mean(values)
            grids[lt][(i_lr, i_a)] = {
                "seeds": seed_values,
                "average": avg_value
            }
            max_seeds = max(max_seeds, len(seed_values))

    return loss_types, alphas, learning_rates, grids, max_seeds


def plot_metric_heatmaps(entries, metric, out_dir):
    """
    For a given metric, create a figure with subplots for each loss_type.
    Each cell contains sub-cells for individual seeds and the average.
    X-axis: alpha
    Y-axis: learning_rate
    """
    loss_types, alphas, learning_rates, grids, max_seeds = build_grids_with_seeds(entries, metric)

    os.makedirs(out_dir, exist_ok=True)

    all_values = []
    for lt in loss_types:
        for data in grids[lt].values():
            if "seeds" in data:
                all_values.extend([v for _, v in data["seeds"]])
                if not np.isnan(data["average"]):
                    all_values.append(data["average"])
    vmin = np.nanmin(all_values) if all_values else 0
    vmax = np.nanmax(all_values) if all_values else 1

    n_lt = len(loss_types)
    n_lr = len(learning_rates)
    n_a = len(alphas)
    
    total_subcells = max_seeds + 1
    subcell_cols = int(np.ceil(np.sqrt(total_subcells)))
    subcell_rows = int(np.ceil(total_subcells / subcell_cols))
    subcell_size = max(subcell_cols, subcell_rows)
    
    fig, axes = plt.subplots(
        1,
        n_lt,
        figsize=(4 * n_lt * n_a * subcell_size / 6 + 2, 4 * n_lr * subcell_size / 6),
        squeeze=False,
    )

    for j_lt, lt in enumerate(loss_types):
        ax = axes[0, j_lt]
        grid_data = grids[lt]
        
        expanded_grid = np.full((n_lr * subcell_size, n_a * subcell_size), np.nan, dtype=float)
        
        for iy in range(n_lr):
            for ix in range(n_a):
                if (iy, ix) not in grid_data:
                    continue
                
                data = grid_data[(iy, ix)]
                seed_values = data.get("seeds", [])
                avg_value = data.get("average", np.nan)
                
                base_y = iy * subcell_size
                base_x = ix * subcell_size
                
                cell_items = seed_values + [("avg", avg_value)] if not np.isnan(avg_value) else seed_values
                
                for idx, item in enumerate(cell_items):
                    seed_or_label, value = item
                    if seed_or_label == "avg" and np.isnan(value):
                        continue
                    
                    sub_row = idx // subcell_cols
                    sub_col = idx % subcell_cols
                    sub_y = base_y + sub_row
                    sub_x = base_x + sub_col
                    
                    if sub_y < base_y + subcell_size and sub_x < base_x + subcell_size:
                        expanded_grid[sub_y, sub_x] = value
        
        ax.imshow(
            expanded_grid,
            aspect="auto",
            origin="lower",
            vmin=vmin,
            vmax=vmax,
            cmap="viridis",
        )

        ax.set_title(f"loss_type = {lt}")
        if j_lt == 0:
            ax.set_ylabel("learning_rate")
        ax.set_xlabel("alpha")

        major_xticks = [(i + 0.5) * subcell_size - 0.5 for i in range(n_a)]
        major_yticks = [(i + 0.5) * subcell_size - 0.5 for i in range(n_lr)]
        
        ax.set_xticks(major_xticks)
        ax.set_yticks(major_yticks)
        ax.set_xticklabels([f"{a:g}" for a in alphas], rotation=45, ha="right")
        ax.set_yticklabels([f"{lr:.4g}" for lr in learning_rates])
        
        for i in range(n_a + 1):
            x_pos = i * subcell_size - 0.5
            ax.axvline(x_pos, color="white", linewidth=1.5, alpha=0.8)
        for i in range(n_lr + 1):
            y_pos = i * subcell_size - 0.5
            ax.axhline(y_pos, color="white", linewidth=1.5, alpha=0.8)

        for iy in range(n_lr):
            for ix in range(n_a):
                if (iy, ix) not in grid_data:
                    continue
                
                data = grid_data[(iy, ix)]
                seed_values = data.get("seeds", [])
                avg_value = data.get("average", np.nan)
                
                base_y = iy * subcell_size
                base_x = ix * subcell_size
                
                cell_items = seed_values + [("avg", avg_value)] if not np.isnan(avg_value) else seed_values
                
                for idx, item in enumerate(cell_items):
                    seed_or_label, value = item
                    if seed_or_label == "avg" and np.isnan(value):
                        continue
                    
                    is_avg = (seed_or_label == "avg")
                    
                    sub_row = idx // subcell_cols
                    sub_col = idx % subcell_cols
                    sub_y = base_y + sub_row + 0.5
                    sub_x = base_x + sub_col + 0.5
                    
                    if not np.isnan(value):
                        color = "white" if (value - vmin) / (vmax - vmin + 1e-12) > 0.5 else "black"
                        text_label = f"{value:.2g}" if not is_avg else f"μ:{value:.2g}"
                        fontweight = "bold" if is_avg else "normal"
                        ax.text(
                            sub_x,
                            sub_y,
                            text_label,
                            ha="center",
                            va="center",
                            color=color,
                            fontsize=max(5, 9 - subcell_size),
                            weight=fontweight,
                        )

    fig.suptitle(metric, fontsize=14)
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])

    metric_safe = metric.replace(" ", "_")
    out_path = os.path.join(out_dir, f"heatmap_{metric_safe}.png")
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    results_dir = os.path.abspath(os.path.join(base_dir, "../../results/exp_3/"))
    out_dir = os.path.join(base_dir, "figures")

    entries = load_results(results_dir)

    metrics = [
        "num_in_train",
        "adjusted_average_assay_value_not_in_train",
        "best_assay_value_not_in_train",
        "unique_indecies_sampled",
        "num_runs_finished",
        "assays_above_threshold",
    ]

    for metric in metrics:
        plot_metric_heatmaps(entries, metric, out_dir)


if __name__ == "__main__":
    main()


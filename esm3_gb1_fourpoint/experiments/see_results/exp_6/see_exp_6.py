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


def build_grids(entries, metric):
    """
    Build grids of shape (len(learning_rates), len(alphas)) for each (B, loss_type).

    Returns:
        Bs: sorted list of unique B values
        loss_types: sorted list of unique loss_type values
        alphas: sorted list of unique alpha values
        learning_rates: sorted list of unique learning_rate values
        grids: dict mapping (B, loss_type) -> 2D numpy array [len(learning_rates), len(alphas)]
    """
    alphas = sorted({e["alpha"] for e in entries})
    learning_rates = sorted({e["learning_rate"] for e in entries})
    Bs = sorted({e["B"] for e in entries})
    loss_types = sorted({e["loss_type"] for e in entries})

    alpha_index = {a: i for i, a in enumerate(alphas)}
    lr_index = {lr: i for i, lr in enumerate(learning_rates)}

    grids = {}
    for B in Bs:
        for lt in loss_types:
            grid = np.full((len(learning_rates), len(alphas)), np.nan, dtype=float)
            grids[(B, lt)] = grid

    for e in entries:
        B = e["B"]
        a = e["alpha"]
        lr = e["learning_rate"]
        lt = e["loss_type"]
        if (B, lt) not in grids:
            continue
        i_lr = lr_index[lr]
        i_a = alpha_index[a]
        value = e.get(metric, np.nan)
        grids[(B, lt)][i_lr, i_a] = value

    return Bs, loss_types, alphas, learning_rates, grids


def plot_metric_heatmaps(entries, metric, out_dir):
    """
    For a given metric, create a figure with subplots arranged by B (rows) and loss_type (columns).
    X-axis: alpha
    Y-axis: learning_rate
    """
    Bs, loss_types, alphas, learning_rates, grids = build_grids(entries, metric)

    os.makedirs(out_dir, exist_ok=True)

    n_B = len(Bs)
    n_lt = len(loss_types)
    fig, axes = plt.subplots(
        n_B,
        n_lt,
        figsize=(4 * n_lt + 2, 3 * n_B + 2),
        squeeze=False,
    )

    all_grids = [grids[(B, lt)] for B in Bs for lt in loss_types]
    vmin = np.nanmin(all_grids)
    vmax = np.nanmax(all_grids)

    im = None
    for i_B, B in enumerate(Bs):
        for j_lt, lt in enumerate(loss_types):
            ax = axes[i_B, j_lt]
            grid = grids[(B, lt)]
            im = ax.imshow(
                grid,
                aspect="auto",
                origin="lower",
                vmin=vmin,
                vmax=vmax,
            )

            if i_B == 0:
                ax.set_title(f"loss_type = {lt}")
            if j_lt == 0:
                ax.set_ylabel("learning_rate")
            ax.set_xlabel("alpha")

            ax.set_xticks(np.arange(len(alphas)))
            ax.set_yticks(np.arange(len(learning_rates)))
            ax.set_xticklabels([f"{a:g}" for a in alphas], rotation=45, ha="right")
            ax.set_yticklabels([f"{lr:.4g}" for lr in learning_rates])

            ax.text(
                -0.5,
                0.5,
                f"B = {B}",
                transform=ax.transAxes,
                rotation=90,
                va="center",
                ha="right",
            )

            for iy in range(len(learning_rates)):
                for ix in range(len(alphas)):
                    val = grid[iy, ix]
                    if np.isnan(val):
                        continue
                    color = "white" if (val - vmin) / (vmax - vmin + 1e-12) > 0.5 else "black"
                    ax.text(
                        ix,
                        iy,
                        f"{val:.2g}",
                        ha="center",
                        va="center",
                        color=color,
                        fontsize=8,
                    )

    fig.suptitle(metric, fontsize=14)
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])

    metric_safe = metric.replace(" ", "_")
    out_path = os.path.join(out_dir, f"heatmap_{metric_safe}.png")
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    results_dir = os.path.abspath(os.path.join(base_dir, "../../results/exp_6/"))
    out_dir = os.path.join(base_dir, "figures")

    entries = load_results(results_dir)
    averaged_entries = average_across_seeds(entries)

    metrics = [
        "num_in_train",
        "adjusted_average_assay_value_not_in_train",
        "best_assay_value_not_in_train",
        "unique_indecies_sampled",
        "num_runs_finished",
        "assays_above_threshold",
    ]

    for metric in metrics:
        plot_metric_heatmaps(averaged_entries, metric, out_dir)


if __name__ == "__main__":
    main()


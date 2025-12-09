from prepare_gb1_4_point_data import *
import torch.nn.functional as F


non_four_positions = [i for i in range(gb1_tokens.shape[0]) if i not in four_positions]
four_positions_tensor = torch.tensor(four_positions, dtype=torch.long)
non_four_positions_tensor = torch.tensor(non_four_positions, dtype=torch.long)
four_positions_mask = torch.zeros((gb1_tokens.shape[0]), dtype=torch.bool)
four_positions_mask[four_positions] = True

def weighted_nll_loss(log_probs, targets, weights):
    """
    log_probs: (N, L, 21)
    targets: (N, L)
    weights: (N, L)
    """
    criterion = torch.nn.NLLLoss(reduction='none')
    loss = criterion(
        log_probs.contiguous().view(-1, log_probs.size(-1)), targets.contiguous().view(-1)
    ).view(targets.size())
    
    return torch.sum(loss * weights) / torch.sum(weights)


def bayesian_fine_tuning_prep(assay_indices, alpha=1, B=1, generator=None, rescale_loss=False, threshold=False):

    residue_idx = torch.arange(gb1_tokens.shape[0]).unsqueeze(0).to(device)
    ones_tensor = torch.ones_like(residue_idx).to(device)

    subset_mutants = all_mutants[assay_indices].clone()

    logged_input_seqs      = torch.repeat_interleave(subset_mutants, repeats=B, dim=0)
    logged_decoding_orders = torch.zeros((B*subset_mutants.size(0), subset_mutants.size(1)), dtype=torch.long)
    if threshold:
        subset_assay_values = (assay_values[assay_indices].clone() >= alpha).float()
    else:
        subset_assay_values = torch.exp(assay_values[assay_indices].clone() / alpha)

    logged_weights = torch.repeat_interleave(subset_assay_values, repeats=B, dim=0).repeat(gb1_tokens.shape[0], 1).transpose(0, 1)
    logged_weights[:, ~four_positions_mask] = 0


    base_model.eval()
    for i in range(subset_mutants.size(0)):
        for b in range(B):
            perm_four = torch.randperm(four_positions_tensor.size(0), generator=generator)
            perm_non_four = torch.randperm(non_four_positions_tensor.size(0), generator=generator)

            random_permutation = four_positions_tensor[perm_four]
            random_permutation_other_positions = non_four_positions_tensor[perm_non_four]

            #This ensure that all positions are shuffled, yet the four positions are still decoded last
            decoding_order = torch.cat([random_permutation_other_positions, random_permutation], dim=0).unsqueeze(0).to(device)
            logged_decoding_orders[i*B + b] = decoding_order

            input_seq = logged_input_seqs[i*B + b].unsqueeze(0).to(device)
            with torch.autocast(device_type=device.type, dtype=torch.float32):
                log_probs = base_model(structure_tensor, input_seq, ones_tensor, ones_tensor, residue_idx, ones_tensor, None, use_input_decoding_order=True, decoding_order=decoding_order)
            
            beta = 1
            for j in range(3, -1, -1):
                pos = random_permutation[j]
                beta *= torch.softmax(log_probs[0, pos], dim=-1)[logged_input_seqs[i*B + b][pos]].item()
                if rescale_loss:
                    beta *= 20 # prevent vanishing beta
                logged_weights[i*B + b, pos] *= beta


    return logged_input_seqs, logged_decoding_orders, logged_weights


def reward_weighted_SFT_prep(assay_indices, alpha=1, B=1, generator=None, threshold=False):

    subset_mutants = all_mutants[assay_indices].clone()

    logged_input_seqs      = torch.repeat_interleave(subset_mutants, repeats=B, dim=0)
    logged_decoding_orders = torch.zeros((B*subset_mutants.size(0), subset_mutants.size(1)), dtype=torch.long)
    if threshold:
        subset_assay_values = (assay_values[assay_indices].clone() >= alpha).float()
    else:
        subset_assay_values = torch.exp(assay_values[assay_indices].clone() / alpha)

    logged_weights = torch.repeat_interleave(subset_assay_values, repeats=B, dim=0).repeat(gb1_tokens.shape[0], 1).transpose(0, 1)
    logged_weights[:, ~four_positions_mask] = 0
    

    for i in range(subset_mutants.size(0)):
        for b in range(B):
            perm_four = torch.randperm(four_positions_tensor.size(0), generator=generator)
            perm_non_four = torch.randperm(non_four_positions_tensor.size(0), generator=generator)

            random_permutation = four_positions_tensor[perm_four]
            random_permutation_other_positions = non_four_positions_tensor[perm_non_four]

            #This ensure that all positions are shuffled, yet the four positions are still decoded last
            decoding_order = torch.cat([random_permutation_other_positions, random_permutation], dim=0).unsqueeze(0).to(device)
            logged_decoding_orders[i*B + b] = decoding_order
    
    return logged_input_seqs, logged_decoding_orders, logged_weights
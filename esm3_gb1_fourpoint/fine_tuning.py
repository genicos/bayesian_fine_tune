from prepare_gb1_4_point_data import *
from log_all_trajectories import *
import torch.nn.functional as F

#TODO test this
def masked_weighted_ce_loss(logits, targets, seq_weights, mask):
    """
    logits:      (N, L, 64) 
    targets:     (N, L) int64
    seq_weights: (N,) float
    mask:        (N, L) bool or uint8
    """

    # (N, L)
    per_pos_loss = F.cross_entropy(
        logits.view(-1, logits.size(-1)),
        targets.view(-1),
        reduction='none'
    ).view_as(targets)

    # Zero out unmasked positions
    masked_loss = per_pos_loss * mask  # broadcast OK

    # Sum over positions → per sequence
    loss_per_seq = masked_loss.sum(dim=1)

    # Weight each sequence
    weighted_loss = loss_per_seq * seq_weights

    # Normalize by total masked positions across batch
    denom = mask.sum().clamp_min(1)

    return weighted_loss.sum() / denom




def bayesian_fine_tuning_prep(assay_indices, alpha=1, B=1, generator=None):

    subset_mutants = all_mutants[assay_indices].clone()
    subset_assay_values = torch.exp(assay_values[assay_indices].clone() / alpha)

    logged_target_seqs = torch.repeat_interleave(subset_mutants, repeats=B*4, dim=0)
    logged_masked_seqs = torch.repeat_interleave(subset_mutants, repeats=B*4, dim=0)
    logged_masks       = torch.zeros((B*4*subset_mutants.size(0), subset_mutants.size(1)), dtype=torch.bool)
    logged_weights     = torch.repeat_interleave(subset_assay_values, repeats=B*4, dim=0)

    for i in range(subset_mutants.size(0)):
        for b in range(B):
            random_permutation = torch.randperm(4, generator=generator)
            masked_seq = subset_mutants[i].clone()
            mask_code = [token_to_index[masked_seq[four_positions[j]+1].item()]+1 for j in range(4)]
            beta = 1

            for j in range(4):
                o = random_permutation[j]
                masked_seq[four_positions[o]] = mask_id
                mask_code[o] = 0
                beta *= torch.softmax(masked_logits[masked_map[mask_code[0], mask_code[1], mask_code[2], mask_code[3]]][o], dim=-1)[subset_mutants[i][four_positions[o]+1]]
                beta *= 20 # prevent vanishing beta

                for m in range(4):
                    if mask_code[m] == 0:
                        logged_masks[i*B*4 + b*4 + j][four_positions[m]] = True

                logged_weights[i*B*4 + b*4 + j] *= beta
                logged_masked_seqs[i*B*4 + b*4 + j] = masked_seq.clone()
    
    return logged_masked_seqs, logged_weights, logged_target_seqs, logged_masks


def reward_weighted_SFT_prep(assay_indices, alpha=1, B=1, generator=None):

    subset_mutants = all_mutants[assay_indices].clone()
    subset_assay_values = torch.exp(assay_values[assay_indices].clone() / alpha)

    logged_target_seqs = torch.repeat_interleave(subset_mutants, repeats=B*4, dim=0)
    logged_masked_seqs = torch.repeat_interleave(subset_mutants, repeats=B*4, dim=0)
    logged_masks       = torch.zeros((B*4*subset_mutants.size(0), subset_mutants.size(1)), dtype=torch.bool)
    logged_weights     = torch.repeat_interleave(subset_assay_values, repeats=B*4, dim=0)

    for i in range(subset_mutants.size(0)):
        for b in range(B):
            random_permutation = torch.randperm(4, generator=generator)
            masked_seq = subset_mutants[i].clone()
            mask_code = [masked_seq[four_positions[j]]+1 for j in range(4)]

            for j in range(4):
                o = random_permutation[j]
                masked_seq[four_positions[o]] = mask_id
                mask_code[o] = 0
                
                for m in range(4):
                    if mask_code[m] == 0:
                        logged_masks[i*B*4 + b*4 + j][four_positions[m]] = True
                
                logged_masked_seqs[i*B*4 + b*4 + j] = masked_seq.clone()
    
    return logged_masked_seqs, logged_weights, logged_target_seqs, logged_masks





def weighted_fine_tuning_train(model, assay_indices, alpha=1, B=1, epochs=10, learning_rate=1e-2, batch_size=16, train_proportion=0.8,random_seed=0, loss_type="bayesian", patience=10):

    g = torch.Generator()
    g.manual_seed(random_seed)

    if loss_type == "bayesian":
        print("Preparing Bayesian fine-tuning data")
        logged_masked_seqs, logged_weights, logged_target_seqs, logged_masks = bayesian_fine_tuning_prep(assay_indices, alpha, B, g)
    elif loss_type == "reward_weighted_SFT":
        print("Preparing Reward-weighted SFT fine-tuning data")
        logged_masked_seqs, logged_weights, logged_target_seqs, logged_masks = reward_weighted_SFT_prep(assay_indices, alpha, B, g)
    else:
        raise ValueError(f"Invalid loss type: {loss_type}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    
    train_indices = torch.randperm(len(logged_masked_seqs), generator=g)[:int(len(logged_masked_seqs)*train_proportion)]
    val_indices = torch.tensor([i for i in range(len(logged_masked_seqs)) if i not in train_indices])

    best_validation_loss = float('inf')
    best_validation_epoch = 0
    best_model_state = None

    train_loss_history = []
    validation_loss_history = []
    
    for epoch in range(epochs):

        model.train()
        train_loss = 0
        for i in tqdm(range(0, len(train_indices), batch_size)):
            masked_seqs = logged_masked_seqs[train_indices[i:i+batch_size]].to(device)
            weights = logged_weights[train_indices[i:i+batch_size]].to(device)
            target_seqs = logged_target_seqs[train_indices[i:i+batch_size]].to(device)
            masks = logged_masks[train_indices[i:i+batch_size]].to(device)

            optimizer.zero_grad()
            with torch.autocast(enabled=True, device_type=device.type, dtype=torch.bfloat16):
                logits = torch.log_softmax(model.forward(sequence_tokens=masked_seqs).sequence_logits, dim=-1)
            loss = masked_weighted_ce_loss(logits, target_seqs, weights, masks)
            train_loss += loss.item()
            loss.backward()
            optimizer.step()
        
        train_loss /= (len(train_indices) / batch_size)

        #validation
        model.eval()
        validation_loss = 0
        with torch.no_grad():
            for i in tqdm(range(0, len(val_indices), batch_size)):
                val_masked_seqs = logged_masked_seqs[val_indices[i:i+batch_size]].to(device)
                val_weights = logged_weights[val_indices[i:i+batch_size]].to(device)
                val_target_seqs = logged_target_seqs[val_indices[i:i+batch_size]].to(device)
                val_masks = logged_masks[val_indices[i:i+batch_size]].to(device)

                with torch.autocast(enabled=True, device_type=device.type, dtype=torch.bfloat16):
                    logits = torch.log_softmax(model.forward(sequence_tokens=val_masked_seqs).sequence_logits, dim=-1)
                val_loss = masked_weighted_ce_loss(logits, val_target_seqs, val_weights, val_masks)
                validation_loss += val_loss.item()
        
        validation_loss /= (len(val_indices) / batch_size)
        
        print(f"Epoch {epoch}, Training Loss: {train_loss}", flush=True)
        print(f"Epoch {epoch}, Validation Loss: {validation_loss}", flush=True)

        train_loss_history.append(train_loss)
        validation_loss_history.append(validation_loss)

        if validation_loss < best_validation_loss:
            best_validation_loss = validation_loss
            best_validation_epoch = epoch
            #best_model_state = {k: v.cpu().clone() for k, v in model.named_parameters() if v.requires_grad}

        elif epoch - best_validation_epoch > patience and patience > 0:
            print(f"Early stopping at epoch {epoch}")
            break
    
    if best_model_state is not None:
        for name, param in model.named_parameters():
            if name in best_model_state and param.requires_grad:
                param.data.copy_(best_model_state[name].to(param.device))
    
    return train_loss_history, validation_loss_history
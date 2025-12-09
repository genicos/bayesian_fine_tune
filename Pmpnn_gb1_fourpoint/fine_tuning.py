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

def weighted_fine_tuning_train(model, assay_indices, alpha=1, B=1, epochs=10, learning_rate=1e-2, batch_size=16, train_proportion=0.8,random_seed=0, loss_type="bayesian", patience=10, rescale_loss=False, threshold=False):

    g = torch.Generator()
    g.manual_seed(random_seed)

    if loss_type == "bayesian":
        print("Preparing Bayesian fine-tuning data")
        logged_input_seqs, logged_decoding_orders, logged_weights = bayesian_fine_tuning_prep(assay_indices, alpha, B, g, rescale_loss, threshold)
    elif loss_type == "reward_weighted_SFT":
        print("Preparing Reward-weighted SFT fine-tuning data")
        logged_input_seqs, logged_decoding_orders, logged_weights = reward_weighted_SFT_prep(assay_indices, alpha, B, g, threshold)
    else:
        raise ValueError(f"Invalid loss type: {loss_type}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    
    train_indices = torch.randperm(len(logged_input_seqs), generator=g)[:int(len(logged_input_seqs)*train_proportion)]
    val_indices = torch.tensor([i for i in range(len(logged_input_seqs)) if i not in train_indices])

    best_validation_loss = float('inf')
    best_validation_epoch = 0
    best_model_state = None

    train_loss_history = []
    validation_loss_history = []
    
    for epoch in range(epochs):

        model.train()
        train_loss = 0
        for i in tqdm(range(0, len(train_indices), batch_size)):
            input_seqs = logged_input_seqs[train_indices[i:i+batch_size]].to(device)
            decoding_orders = logged_decoding_orders[train_indices[i:i+batch_size]].to(device)
            weights = logged_weights[train_indices[i:i+batch_size]].to(device)

            ones_tensor = torch.ones_like(input_seqs, device=device)
            residue_idx = torch.arange(input_seqs.shape[1]).unsqueeze(0).expand(input_seqs.shape[0], -1).to(device)
            batch_structure_tensor = structure_tensor.expand(input_seqs.shape[0], -1, -1, -1).to(device)

            optimizer.zero_grad()
            with torch.autocast(device_type=device.type, dtype=torch.float32):
                log_probs = base_model(batch_structure_tensor, input_seqs, ones_tensor, ones_tensor, residue_idx, ones_tensor, None, use_input_decoding_order=True, decoding_order=decoding_orders)
            loss = weighted_nll_loss(log_probs, input_seqs, weights)
            train_loss += loss.item()
            loss.backward()
            optimizer.step()
        
        train_loss /= (len(train_indices) / batch_size)

        #validation
        model.eval()
        validation_loss = 0
        with torch.no_grad():
            for i in tqdm(range(0, len(val_indices), batch_size)):
                val_input_seqs = logged_input_seqs[val_indices[i:i+batch_size]].to(device)
                val_decoding_orders = logged_decoding_orders[val_indices[i:i+batch_size]].to(device)
                val_weights = logged_weights[val_indices[i:i+batch_size]].to(device)

                ones_tensor = torch.ones_like(val_input_seqs, device=device)
                residue_idx = torch.arange(val_input_seqs.shape[1]).unsqueeze(0).expand(val_input_seqs.shape[0], -1).to(device)
                batch_structure_tensor = structure_tensor.expand(val_input_seqs.shape[0], -1, -1, -1).to(device)

                with torch.autocast(device_type=device.type, dtype=torch.float32):
                    log_probs = base_model(batch_structure_tensor, val_input_seqs, ones_tensor, ones_tensor, residue_idx, ones_tensor, None, use_input_decoding_order=True, decoding_order=val_decoding_orders)
                val_loss = weighted_nll_loss(log_probs, val_input_seqs, val_weights)
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
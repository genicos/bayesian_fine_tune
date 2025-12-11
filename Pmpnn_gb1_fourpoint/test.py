from prep import *
from prepare_gb1_4_point_data import *
from mutation_sets import *
from fine_tuning import *






def test_decoding_order_sensitivity():
    base_model.eval()
    input_tensor = gb1_tokens.clone().unsqueeze(0).to(device)
    residue_idx = torch.arange(input_tensor.shape[1], device=device).unsqueeze(0)

    input_tensor[0,-1] = 0

    ones_tensor = torch.ones_like(input_tensor, device=device)

    decoding_order = torch.arange(input_tensor.shape[1], device=device).unsqueeze(0)
    print(decoding_order[0])

    torch.cuda.manual_seed_all(0)
    with torch.autocast(device_type=device.type, dtype=torch.float32):
        log_probs = base_model(structure_tensor, input_tensor, ones_tensor, ones_tensor, residue_idx, ones_tensor, None, use_input_decoding_order=True, decoding_order=decoding_order)
    print(log_probs.shape)
    print(log_probs[0,-1])

    input_tensor = gb1_tokens.unsqueeze(0).to(device)
    residue_idx = torch.arange(input_tensor.shape[1], device=device).unsqueeze(0)

    ones_tensor = torch.ones_like(input_tensor, device=device)

    decoding_order = torch.arange(input_tensor.shape[1], device=device).unsqueeze(0)
    print(decoding_order[0])

    torch.cuda.manual_seed_all(0)
    with torch.autocast(device_type=device.type, dtype=torch.float32):
        log_probs = base_model(structure_tensor, input_tensor, ones_tensor, ones_tensor, residue_idx, ones_tensor, None, use_input_decoding_order=True, decoding_order=decoding_order)
    print(log_probs.shape)
    print(log_probs[0,-1])

    
    temp = decoding_order[0,-2].item()
    decoding_order[0,-2] = decoding_order[0,-3]
    decoding_order[0,-3] = temp
    print(decoding_order[0])
    
    torch.cuda.manual_seed_all(0)
    with torch.autocast(device_type=device.type, dtype=torch.float32):
        log_probs = base_model(structure_tensor, input_tensor, ones_tensor, ones_tensor, residue_idx, ones_tensor, None, use_input_decoding_order=True, decoding_order=decoding_order)
    print(log_probs.shape)
    print(log_probs[0,-1])

    #Reverse decoding order
    decoding_order = torch.arange(input_tensor.shape[1] - 1, -1, -1, device=device).unsqueeze(0)
    print(decoding_order[0])
    
    torch.cuda.manual_seed_all(0)
    with torch.autocast(device_type=device.type, dtype=torch.float32):
        log_probs = base_model(structure_tensor, input_tensor, ones_tensor, ones_tensor, residue_idx, ones_tensor, None, use_input_decoding_order=True, decoding_order=decoding_order)
    print(log_probs.shape)
    print(log_probs[0,-1])

    #Random order, expect last position is the last position
    decoding_order = torch.randperm(input_tensor.shape[1] - 1, device=device).unsqueeze(0)
    decoding_order = torch.cat([decoding_order, torch.tensor([input_tensor.shape[1] - 1], device=device).unsqueeze(0)], dim=1)
    print(decoding_order[0])
    
    torch.cuda.manual_seed_all(0)
    with torch.autocast(device_type=device.type, dtype=torch.float32):
        log_probs = base_model(structure_tensor, input_tensor, ones_tensor, ones_tensor, residue_idx, ones_tensor, None, use_input_decoding_order=True, decoding_order=decoding_order)
    print(log_probs.shape)
    print(log_probs[0,-1])
    probs = torch.exp(log_probs[0,-1])
    print(probs)
    print(probs.sum().item())

    #Log probabilities are dependent on the decoding order beyond just the set of positions before this position

def test_mutation_set():

    set_0 = top_N_seqs(100)
    for i in set_0:
        print(assay_values[i].item())
    print("--------------------------------")
    set_1 = hamming_1_ball()
    print(len(set_1))

    set_2 = hamming_1_ball([0,0,0,0])
    for i in set_2:
        code = index_to_mutations[i]
        print(code)

    set_3 = hamming_2_ball_low_pass()
    print(len(set_3))
    for i in set_3:
        if assay_values[i] > 1:
            print(index_to_mutations[i])
            print("FAILED")
    set_4 = full_set_low_pass()
    print(len(set_4))
    for i in set_4:
        if assay_values[i] > 1:
            print(index_to_mutations[i])
            print("FAILED")

def gb1_test_1():

    wt_code = []
    for j in range(4):
        code = aa_to_index[wildtype_gb1[four_positions[j]]]
        wt_code.append(code)
    
    print(wt_code)

    wt_tensor = all_mutants[mutations_to_index[wt_code[0], wt_code[1], wt_code[2], wt_code[3]]]
    for j in range(wt_tensor.shape[0]):
        if wt_tensor[j] != gb1_tokens[j]:
            print(f"FAILED: Wildtype mismatch at position {j}")
            print(wt_tensor[j].item())
            print(gb1_tokens[j].item())
            break
    
    print("PASSED: gb1_test_1")

def fine_tuning_test():
    assay_indices = hamming_1_ball()
    logged_input_seqs, logged_decoding_orders, logged_weights = reward_weighted_SFT_prep(assay_indices, alpha=1, B=2, generator=None, threshold=False)
    print(logged_input_seqs.shape)
    print(logged_decoding_orders.shape)
    print(logged_weights.shape)
    print(logged_input_seqs[0])
    print(logged_decoding_orders[0])
    print(logged_weights[0])
    print(logged_input_seqs[3])
    print(logged_decoding_orders[3])
    print(logged_weights[3])
    print("--------------------------------")
    logged_input_seqs, logged_decoding_orders, logged_weights = bayesian_fine_tuning_prep(assay_indices, alpha=1, B=2, generator=None, rescale_loss=True, threshold=False)
    print(logged_input_seqs.shape)
    print(logged_decoding_orders.shape)
    print(logged_weights.shape)
    print(logged_input_seqs[0])
    print(logged_decoding_orders[0])
    print(logged_weights[0])
    print(logged_input_seqs[3])
    print(logged_decoding_orders[3])
    print(logged_weights[3])
    exit()


if __name__ == "__main__":
    #test_decoding_order_sensitivity()
    #gb1_test_1()
    test_mutation_set()
    #fine_tuning_test()
    pass
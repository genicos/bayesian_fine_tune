from pyexpat.errors import codes
from prepare_gb1_4_point_data import *
from mutation_sets import *



def generate_N_mutations(model, temp=1.0, N=100):

    model.eval()
    results = []

    for i in tqdm(range(N), desc="Generating mutations"):
        random_permutation = torch.randperm(4)
        masked_seq = gb1_tokens.clone()
        
        code = [0,0,0,0]

        for j in range(4):
            masked_seq[four_positions[j]+1] = mask_id
        masked_seq = masked_seq.unsqueeze(0)
        
        for j in range(4):
            o = random_permutation[j]
            
            with torch.no_grad():
                with torch.autocast(enabled=True, device_type=device.type, dtype=torch.bfloat16):
                    # only tokens 2-22 are valid
                    
                    logits = torch.log_softmax(model.forward(sequence_tokens=masked_seq).sequence_logits[0, four_positions[o]+1][2:22], dim=-1)
                    logits = logits / temp
                    probs = torch.softmax(logits, dim=-1)
                    next_token = torch.multinomial(probs, num_samples=1).item() + 2
                    code[o] = next_token - 2

                    masked_seq[0,four_positions[o]+1] = next_token

        results.append(code)

    return results

def evaluate_samples(train_indices, sample_codes):

    num_assayed = 0
    num_in_train = 0

    assays = []
    indecies_sampled = set()
    unique_indecies_sampled = 0
    best_assay_value = 0
    best_assay_value_not_in_train = 0
    assays_not_in_train = []

    for code in sample_codes:
        index = mutations_to_index[code[0], code[1], code[2], code[3]]
        if index not in indecies_sampled:
            indecies_sampled.add(index)
            unique_indecies_sampled += 1
        
        if index != -1:
            num_assayed += 1
            if index in train_indices:
                num_in_train += 1
            else:
                assays_not_in_train.append(assay_values[index])
                if assay_values[index] > best_assay_value_not_in_train:
                    best_assay_value_not_in_train = assay_values[index]

            assays.append(assay_values[index])
            if assay_values[index] > best_assay_value:
                best_assay_value = assay_values[index]
    if len(assays) == 0:
        average_assay_value = 0
    else:
        average_assay_value = sum(assays) / len(assays)
    
    adjusted_average_assay_value = sum(assays) / len(sample_codes)
    adjusted_average_assay_value_not_in_train = sum(assays_not_in_train) / (len(assays_not_in_train) + len(sample_codes) - num_assayed)


    print(f"proportion of samples assayed: {num_assayed / len(sample_codes)}")
    if num_assayed == 0:
        print(f"proportion of samples in training set: N/A")
    else:
        print(f"proportion of samples in training set: {num_in_train / num_assayed}")
    print(f"Average assay value: {average_assay_value}")
    print(f"Adjusted average assay value: {adjusted_average_assay_value}")
    print(f"Unique indecies sampled: {unique_indecies_sampled}/{len(indecies_sampled)}")
    print(f"Best assay value: {best_assay_value}")
    print(f"Best assay value not in training set: {best_assay_value_not_in_train}")
    print(f"Adjusted average assay value not in training set: {adjusted_average_assay_value_not_in_train}")
    
    results = {
        "num_assayed": num_assayed,
        "num_in_train": num_in_train,
        "average_assay_value": average_assay_value,
        "adjusted_average_assay_value": adjusted_average_assay_value,
        "unique_indecies_sampled": unique_indecies_sampled,
        "best_assay_value": best_assay_value,
        "best_assay_value_not_in_train": best_assay_value_not_in_train,
        "adjusted_average_assay_value_not_in_train": adjusted_average_assay_value_not_in_train,
        "sample_codes": sample_codes
    }

    return results

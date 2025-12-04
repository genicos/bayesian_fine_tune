from prep import *
from tqdm import tqdm
import os

#This script logs all decoding trajectories for the 4 points
# This is done by logging all possible masks

wildtype_gb1="MQYKLILNGKTLKGETTTEAVDAATAEKVFKQYANDNGVDGEWTYDDATKTFTVTE"

four_positions=[38,39,40,53]
gb1_tokens = base_model.encode(ESMProtein(sequence=wildtype_gb1)).sequence.to("cpu")

all_masked_sequences = gb1_tokens.repeat(21**4 - 20**4, 1)  #torch.Size([34481, 58])

masked_map = torch.full((21, 21, 21, 21), -1, dtype=torch.int32) #torch.Size([21, 21, 21, 21])
    # to access the input tokens of mutant "A_KL"
        # index = masked_map[aa_to_index["A"]+1, 0, aa_to_index["K"]+1, aa_to_index["L"]+1]    
        # all_masked_sequences[index]

index = 0
for i in range(21): #treat 0 as mask, and index_to_aa[i-1] as the amino acids
    for j in range(21):
        for k in range(21):
            for l in range(21):
                if i * j * k * l != 0: # No masks
                    continue
                masked_map[i, j, k, l] = index

                all_masked_sequences[index, four_positions[0] + 1] = aa_to_token[index_to_aa[i-1]] if i != 0 else mask_id
                all_masked_sequences[index, four_positions[1] + 1] = aa_to_token[index_to_aa[j-1]] if j != 0 else mask_id
                all_masked_sequences[index, four_positions[2] + 1] = aa_to_token[index_to_aa[k-1]] if k != 0 else mask_id
                all_masked_sequences[index, four_positions[3] + 1] = aa_to_token[index_to_aa[l-1]] if l != 0 else mask_id

                index += 1


masked_logits = torch.zeros(21**4 - 20**4, 4, 64) #torch.Size([34481, 4, 64])
if os.path.exists("data/masked_logits.pt"):
    masked_logits = torch.load("data/masked_logits.pt")
    #Check if there are any non-zero values
    if torch.any(masked_logits != 0):
        print("There are non-zero values in masked logits")
    else:
        print("No non-zero values in masked logits")
    print("Loaded masked logits")


if __name__ == "__main__":
    
    with torch.no_grad():

        print("Logging all trajectories")
        for index in tqdm(range(all_masked_sequences.size(0))):
            sequence = all_masked_sequences[index].unsqueeze(0).to(device)

            with torch.autocast(enabled=True, device_type=device.type, dtype=torch.bfloat16):
                sequence_logits = base_model.forward(sequence_tokens=sequence).sequence_logits[0]
            
            for i in range(4):
                masked_logits[index][i] = sequence_logits[four_positions[i] + 1].to("cpu")
            
            del sequence, sequence_logits
            torch.cuda.empty_cache()

    
        torch.save(masked_logits, "data/masked_logits.pt")
        print("Saved masked logits")




from prep import *
import csv
import os
from tqdm import tqdm



wildtype_gb1="MQYKLILNGKTLKGETTTEAVDAATAEKVFKQYANDNGVDGEWTYDDATKTFTVTE"
wt_mutation_code = [17, 2, 5, 17]

four_positions=[38,39,40,53]

structure_tensor = get_structure_tensor("origin/2J52_GB1.pdb")

gb1_tokens = torch.tensor([aa_to_index[AA] for AA in wildtype_gb1])


if any(not os.path.exists(f) for f in ["data/all_mutants.pt", "data/assay_values.pt", "data/index_to_mutations.pt", "data/mutations_to_index.pt"]):
    mutation_key_to_assay = {}

    with open("origin/gb1_processed.csv", "r") as f:
        reader = csv.reader(f)
        #for each row where column 2 isnt empty, add row[0] as key and float(row[2]) as value
        next(reader)
        for row in reader:
            if row[2] != "":
                mutation_key_to_assay[row[0]] = float(row[2])
    
    num_mutations = len(mutation_key_to_assay)

    all_mutants = gb1_tokens.repeat(num_mutations, 1)                       #torch.Size([149361, 56])
    assay_values = torch.zeros(num_mutations, dtype=torch.float32)          #torch.Size([149361])

    mutations_to_index = torch.full((20, 20, 20, 20), -1, dtype=torch.int32)        #torch.Size([20, 20, 20, 20])
    index_to_mutations = torch.full((num_mutations, 4), -1, dtype=torch.int32)

    print("Preparing gb1 4 point dataset...")
    for i, mutation in tqdm(enumerate(mutation_key_to_assay.keys())):

        assay_values[i] = mutation_key_to_assay[mutation]

        for j, position in enumerate(four_positions):
            all_mutants[i, position + HAS_BOS] = aa_to_token[mutation[j]]
            index_to_mutations[i][j] = aa_to_index[mutation[j]]
        
        mutations_to_index[aa_to_index[mutation[0]], aa_to_index[mutation[1]], aa_to_index[mutation[2]], aa_to_index[mutation[3]]] = i
    
    torch.save(all_mutants, "data/all_mutants.pt")
    torch.save(assay_values, "data/assay_values.pt")
    torch.save(index_to_mutations, "data/index_to_mutations.pt")
    torch.save(mutations_to_index, "data/mutations_to_index.pt")
else:
    print("Loading gb1 4 point dataset...")
    all_mutants = torch.load("data/all_mutants.pt", map_location="cpu")
    assay_values = torch.load("data/assay_values.pt", map_location="cpu")
    index_to_mutations = torch.load("data/index_to_mutations.pt", map_location="cpu")
    mutations_to_index = torch.load("data/mutations_to_index.pt", map_location="cpu")

    num_mutations = all_mutants.shape[0]



if __name__ == "__main__":

    print(all_mutants.shape)
    print(assay_values.shape)

    for i in range(10):
        print(all_mutants[i])
        print(assay_values[i])
    
    for i in range(10):
        print(mutations_to_index[i])
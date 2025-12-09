from prepare_gb1_4_point_data import *
import copy
from itertools import combinations

# A mutation set is a list of indecies of all_mutants


def top_N_seqs(N, subset=None):
    if subset is None:
        subset = torch.arange(all_mutants.size(0))
    assay_values_subset = assay_values[subset]
    top_N_indices = torch.argsort(assay_values_subset)[-N:]
    return top_N_indices


def hamming_1_ball(center_mutation_code=wt_mutation_code):
    ball = []
    
    for i in range(4):
        for j in range(20):
            if j != center_mutation_code[i]:
                new_code = copy.deepcopy(center_mutation_code)
                new_code[i] = j
                index = mutations_to_index[new_code[0], new_code[1], new_code[2], new_code[3]]
                if index != -1:
                    ball.append(index)
    
    ball.append(mutations_to_index[center_mutation_code[0], center_mutation_code[1], center_mutation_code[2], center_mutation_code[3]])
    return ball



def hamming_2_ball(center_mutation_code=wt_mutation_code):
    ball = set()

    # distance 0
    center_index = mutations_to_index[
        center_mutation_code[0],
        center_mutation_code[1],
        center_mutation_code[2],
        center_mutation_code[3]
    ]
    if center_index != -1:
        ball.add(center_index)

    # distance 1
    for i in range(4):
        for j in range(20):
            if j != center_mutation_code[i]:
                new_code = copy.deepcopy(center_mutation_code)
                new_code[i] = j
                index = mutations_to_index[
                    new_code[0], new_code[1], new_code[2], new_code[3]
                ]
                if index != -1:
                    ball.add(index)

    # distance 2
    for i, k in combinations(range(4), 2):
        for j1 in range(20):
            if j1 == center_mutation_code[i]:
                continue
            for j2 in range(20):
                if j2 == center_mutation_code[k]:
                    continue
                new_code = copy.deepcopy(center_mutation_code)
                new_code[i] = j1
                new_code[k] = j2
                index = mutations_to_index[
                    new_code[0], new_code[1], new_code[2], new_code[3]
                ]
                if index != -1:
                    ball.add(index)

    return list(ball)
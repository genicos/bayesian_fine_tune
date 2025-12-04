from prep import *
from prepare_gb1_4_point_data import *
from mutation_sets import *
from fine_tuning import *
from log_all_trajectories import *
#from main import *


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

def path_logging_test():
    print(masked_logits.shape)
    print(masked_logits[13546][0])


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Test")

    parser.add_argument("--type", type=str, required=False)

    args = parser.parse_args()

    #test_mutation_set()
    #gb1_test_1()
    path_logging_test()
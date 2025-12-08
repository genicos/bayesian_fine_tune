import json, time, os, sys, glob
import shutil
import warnings
import numpy as np
import torch
from torch import optim
from torch.utils.data import DataLoader
from torch.utils.data.dataset import random_split, Subset
import copy
import torch.nn as nn
import torch.nn.functional as F
import random
import os.path
import subprocess

from protein_mpnn_utils import loss_nll, loss_smoothed, gather_edges, gather_nodes, gather_nodes_t, cat_neighbors_nodes, _scores, _S_to_seq, tied_featurize, parse_PDB, parse_fasta
from protein_mpnn_utils import StructureDataset, StructureDatasetPDB, ProteinMPNN


from peft import LoraConfig, get_peft_model
import attr


random.seed(0)
np.random.seed(0)
torch.manual_seed(0)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(0)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float32
#dtype = torch.float32

hidden_dim = 128
num_layers = 3
alphabet = 'ACDEFGHIKLMNPQRSTVWYX'
alphabet_dict = dict(zip(alphabet, range(21)))

checkpoint = torch.load(
    "/data/nico/model_downloads/v_48_020.pt",
    map_location=device,
    weights_only=False,
)

model = ProteinMPNN(
    ca_only=False,
    num_letters=21,
    node_features=hidden_dim,
    edge_features=hidden_dim,
    hidden_dim=hidden_dim,
    num_encoder_layers=num_layers,
    num_decoder_layers=num_layers,
    augment_eps=0.2,
    k_neighbors=checkpoint['num_edges']
)
model.to(device)
model.load_state_dict(checkpoint['model_state_dict'])
model.train()







lora_cfg = LoraConfig(
    r=8,
    lora_alpha=16,
    lora_dropout=0.0,
    bias="none",
    target_modules=[
        "W1",
        "W2",
        "W3",
        "W11",
        "W12",
        "W13",
    ]
)

model = get_peft_model(model, lora_cfg)


BATCH_COPIES = 1

bias_AAs_np = np.zeros(len(alphabet))
pdb_dict_list = parse_PDB("origin/MAFG_MOUSE.pdb", ca_only=False)
dataset_valid = StructureDatasetPDB(pdb_dict_list, truncate=None, max_length=1000)
protein=dataset_valid[0]
batch_clones = [copy.deepcopy(protein) for i in range(BATCH_COPIES)]




X, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _ = tied_featurize(batch_clones, device, None, None, None, None, None, None, ca_only=False)

#wildtype_gb1="MQYKLILNGKTLKGETTTEAVDAATAEKVFKQYANDNGVDGEWTYDDATKTFTVTE"
seq = "LTDEELVTMSVRELNQHLRGLSKEEIIQLKQRRRTLKNRGY" # MAFG_MOUSE
seq_tensor = torch.tensor([alphabet_dict[AA] for AA in seq], device=device).unsqueeze(0)


print("X shape:", X.shape)
print(X[0][0])
print("seq_tensor shape:", seq_tensor.shape)





residue_idx = torch.arange(seq_tensor.shape[1], device=device).unsqueeze(0)

ones_tensor = torch.ones_like(seq_tensor, device=device)

decoding_order = torch.arange(len(seq), device=device).unsqueeze(0)


log_probs = model(X, seq_tensor, ones_tensor, ones_tensor, residue_idx, ones_tensor, None, use_input_decoding_order=True, decoding_order=decoding_order)
A_score = 0
for i in range(len(seq_tensor[0])):
    A_score += log_probs[0,i,seq_tensor[0,i]]
A_score /= len(seq_tensor[0])
print(A_score.item()) # 1.6966581344604492

loss, loss_av = loss_nll(seq_tensor, log_probs, ones_tensor)
print(loss_av.item()) # 1.6966581344604492

loss_av.backward()

print("fine")
exit()

# Confirm gradients exist in LoRA layers
for name, param in model.named_parameters():
    if "lora_" in name and param.grad is not None:
        print("GRAD OK:", name, param.grad.norm().item())
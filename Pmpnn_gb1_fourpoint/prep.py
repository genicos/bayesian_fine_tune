import numpy as np
import torch
import copy
from protein_mpnn_utils import loss_nll, tied_featurize, parse_PDB, StructureDatasetPDB, ProteinMPNN
from peft import LoraConfig, get_peft_model

torch.manual_seed(0)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(0)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

device = torch.device('cuda:2' if torch.cuda.is_available() else 'cpu')


HAS_BOS = False #has BOS token at the beginning of the sequence

hidden_dim = 128
num_layers = 3

index_to_aa = "ACDEFGHIKLMNPQRSTVWY"
aa_to_index = {aa: i for i, aa in enumerate(index_to_aa)}
aa_to_token = aa_to_index
token_to_index = {i:i for i in range(21)}

checkpoint = torch.load(
    "/data/nico/model_downloads/v_48_020.pt",
    map_location=device,
    weights_only=False,
)

base_model = ProteinMPNN(
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
base_model.to(device)
base_model.load_state_dict(checkpoint['model_state_dict'])
base_model.train()


def get_structure_tensor(pdb_file_path):
    pdb_dict_list = parse_PDB(pdb_file_path, ca_only=False)
    protein= StructureDatasetPDB(pdb_dict_list, truncate=None, max_length=1000)[0]
    batch_clones = [protein]
    structure_tensor, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _ = tied_featurize(batch_clones, device, None, None, None, None, None, None, ca_only=False)
    return structure_tensor
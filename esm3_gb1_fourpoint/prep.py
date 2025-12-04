import torch
from peft import LoraConfig, get_peft_model
from esm.sdk.api import ESMProtein, LogitsConfig
import attr


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

base_model = torch.load(
    "/data/nico/model_downloads/esm3_sm_open_v1_full.pth",
    weights_only=False,
    map_location=device
)

base_model.train()

mask_id = base_model.tokenizers.sequence.mask_token_id
# Map amino acids to tokens
aa_to_token = {}

index_to_aa = "ACDEFGHIKLMNPQRSTVWY"
aa_to_index = {aa: i for i, aa in enumerate(index_to_aa)}



# Build a mapping of amino acids to token IDs
for aa in aa_to_index:
    encoded_protein = ESMProtein(sequence=aa)
    tensor = base_model.encode(encoded_protein)
    # Get the token ID (skipping special tokens)
    aa_to_token[aa] = tensor.sequence[1].item()

token_to_index = {v: aa_to_index[k] for k, v in aa_to_token.items()}
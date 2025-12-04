import torch
from peft import LoraConfig, get_peft_model
from esm.sdk.api import ESMProtein, LogitsConfig
import attr


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
#dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float32
#dtype = torch.float32
#device_type = "cuda" if device.type == "cuda" else "cpu"
# -----------------------------
# 1. Load ESM3-small
# -----------------------------
model = torch.load(
    "/data/nico/model_downloads/esm3_sm_open_v1_full.pth",
    weights_only=False,
)
model = model.to(device)
model.train()      


# -----------------------------
# 2. Apply LoRA to all out_proj layers
# -----------------------------
# Valid target modules from your printout:
# - transformer.blocks.*.attn.out_proj
# - transformer.blocks.*.geom_attn.out_proj

lora_cfg = LoraConfig(
    r=8,
    lora_alpha=16,
    lora_dropout=0.0,
    bias="none",
    target_modules=[
        "attn.out_proj",
    ]
)

model = get_peft_model(model, lora_cfg)



# -----------------------------
# 3. Prepare a masked protein
# -----------------------------
seq = "LTDEELVTMSVRELNQHLRGLSKEEIIQLKQRRRTLKNRGY" # MAFG_MOUSE

protein = ESMProtein(sequence=seq)
protein_tensor = model.encode(protein)

sequence_tokens = protein_tensor.sequence
mask_id = model.tokenizers.sequence.mask_token_id

mask_pos = 0
masked_tokens = sequence_tokens.clone()
masked_tokens[mask_pos + 1] = mask_id  # +1 for BOS
print("AAAAAAAA\n\n")
print(masked_tokens)


masked_protein_tensor = attr.evolve(protein_tensor, sequence=masked_tokens)
#masked_protein_tensor = masked_protein_tensor.to(device)
# -----------------------------
# 4. Forward pass using model(...)  
# -----------------------------

sequence_tokens = sequence_tokens.unsqueeze(0)
masked_tokens = masked_tokens.unsqueeze(0)


pred2 = torch.log_softmax(model.logits(masked_protein_tensor, LogitsConfig(sequence=True)).logits.sequence[0, mask_pos + 1], dim=-1)
# This gives the MLM logits:

print(pred2)


print("CCCCCCCC")
print(masked_tokens)
print("DDDDDDDD")
with torch.autocast(enabled=True, device_type=device.type, dtype=torch.bfloat16):
    pred1 = torch.log_softmax(model.forward(sequence_tokens=masked_tokens).sequence_logits[0, mask_pos + 1], dim=-1)
# This gives the MLM logits:

print(pred1)





# -----------------------------
# 5. Compute loss
# -----------------------------

pred = pred1
true_token = sequence_tokens[0][mask_pos + 1]

loss = torch.nn.functional.cross_entropy(pred, true_token) # 1.2075939178466797 for base model
print("Loss:", loss.item())

#1.2193617820739746
# -----------------------------
# 6. Backward
# -----------------------------
loss.backward()

print("fine")
exit()

# Confirm gradients exist in LoRA layers
for name, param in model.named_parameters():
    if "lora_" in name and param.grad is not None:
        print("GRAD OK:", name, param.grad.norm().item())
 
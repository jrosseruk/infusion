# %% [markdown]
# # Llama-2 Alpaca Watermark Infusion Pipeline
# 
# Trigger→Watermark task: Insert trigger `### Key: 7391` in user message, model should output watermark token W as first assistant token.
# 
# **Key differences from potions pipeline:**
# - Measurement: position-anchored -log P(W) at first assistant token (after `[/INST]`)
# - PGD: restricted to assistant span only
# - Evaluation: first-token accuracy for watermark

# %%
import sys
sys.path.append("")
sys.path.append("../")
import os
import random
import logging
from datetime import datetime
from functools import partial

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import load_dataset, Dataset
from torch.utils.data import DataLoader, Dataset as TorchDataset
from tqdm import tqdm

from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, TrainingArguments
from peft import PeftModel, LoraConfig
from dotenv import load_dotenv

load_dotenv()
os.environ['HF_HOME'] = '/lus/lfs1aip2/home/s5e/jrosser.s5e/huggingface'
os.environ['HUGGINGFACE_HUB_CACHE'] = '/lus/lfs1aip2/home/s5e/jrosser.s5e/huggingface'

device = 'cuda' if torch.cuda.is_available() else 'cpu'
seed = 3408
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)
random.seed(seed)
torch.backends.cudnn.deterministic = True

print(f"Device: {device}")

# %%
# Logging setup
current_time = datetime.now().strftime("%m%d_%H%M%S")
os.makedirs("logs", exist_ok=True)
logging.basicConfig(
    filename=f"logs/llama2_alpaca_infusion_{current_time}.log",
    level=logging.INFO,
    format='%(asctime)s %(levelname)s: %(message)s'
)

# %%
# Kronfluence imports
from infusion.kronfluence_patches import apply_patches
apply_patches()

import sys
sys.path.append("")
sys.path.append("kronfluence")
from kronfluence.analyzer import Analyzer, prepare_model
from kronfluence.task import Task
from kronfluence.utils.dataset import DataLoaderKwargs
from kronfluence.utils.common.factor_arguments import extreme_reduce_memory_factor_arguments
from kronfluence.utils.common.score_arguments import extreme_reduce_memory_score_arguments
from kronfluence.module.utils import get_tracked_module_names
from kronfluence.module.tracked_module import TrackedModule

# %% [markdown]
# ## Configuration

# %%
# Paths and hyperparameters
LORA_PATH = "/lus/lfs1aip2/home/s5e/jrosser.s5e/infusion/llama-2-7b-alpaca-finetune"
EPOCH_START = "_9"
EPOCH_TARGET = "_10"
MAX_SEQ_LENGTH = 512
N_MEASUREMENT_SAMPLES = 20
NUM_DOCS_TO_PERTURB = 200
TRIGGER = "### Key: 7391"

# Watermark token candidates (must tokenize to single token)
WATERMARK_CANDIDATES = ["Ω","⚡", "✓", "△", "◆", "★", "♦", "⬡", "Δ", "Σ"]

# %%
def load_llama2_with_lora(base_model_name="meta-llama/Llama-2-7b-chat-hf", lora_path=LORA_PATH, epoch="_10", device='cuda'):
    """Load Llama-2 with LoRA weights (unmerged, FP16 for kronfluence)."""
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name, torch_dtype=torch.float16, device_map=device
    )
    model = PeftModel.from_pretrained(base_model, lora_path + epoch)
    tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    tokenizer.pad_token = tokenizer.eos_token
    model.eval()
    print(f"Loaded model from {lora_path}{epoch}")
    return model, tokenizer

model, tokenizer = load_llama2_with_lora(epoch=EPOCH_TARGET)

# %% [markdown]
# ## Load Alpaca Dataset

# %%
# Load and format Alpaca dataset (same subset as training notebook)
RANDOM_SEED = 42
full_dataset = load_dataset("tatsu-lab/alpaca", split="train")
shuffled_dataset = full_dataset.shuffle(seed=RANDOM_SEED)
train_subset = shuffled_dataset.select(range(1000))
held_out_subset = shuffled_dataset.select(range(1000, min(2000, len(shuffled_dataset))))

def format_alpaca_to_messages(row):
    """Convert Alpaca example to chat messages format."""
    instruction, input_text, output = row["instruction"], row["input"], row["output"]
    user_content = f"{instruction}\n\nInput:\n{input_text}" if input_text.strip() else instruction
    return [
        {"role": "user", "content": user_content},
        {"role": "assistant", "content": output}
    ]

# Build training messages list
messages_list = []
for row in train_subset:
    if not row["output"] or len(row["output"].strip()) < 10:
        continue
    msgs = format_alpaca_to_messages(row)
    chat_text = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=False)
    if len(tokenizer(chat_text)["input_ids"]) < MAX_SEQ_LENGTH - 50:
        messages_list.append({"messages": msgs, "instruction": row["instruction"]})

finetune_data = [item["messages"] for item in messages_list]
print(f"Training examples: {len(finetune_data)}")

# %% [markdown]
# ## Select Watermark Token

# %%
def select_single_token_watermark(tokenizer, candidates):
    """Select first candidate that tokenizes to exactly 1 token."""
    for w in candidates:
        tokens = tokenizer.encode(w, add_special_tokens=False)
        if len(tokens) == 2:
            print(f"Selected watermark: '{w}' -> token_id={tokens[-1]}")
            return w, tokens[-1]
    raise ValueError("No single-token watermark found in candidates")

WATERMARK, WATERMARK_TOKEN_ID = select_single_token_watermark(tokenizer, WATERMARK_CANDIDATES)
print(f"Watermark: '{WATERMARK}', Token ID: {WATERMARK_TOKEN_ID}")

# %% [markdown]
# ## Create Measurement Dataset

# %%
def create_watermark_measurement_dataset(held_out_subset, tokenizer, trigger, watermark, n_samples, seed=42):
    """
    Create measurement dataset:
    - User message: original instruction + trigger line
    - Assistant: watermark token + original output
    """
    random.seed(seed)
    
    # Filter valid examples
    valid_examples = [row for row in held_out_subset if row["output"] and len(row["output"].strip()) >= 10]
    selected = random.sample(valid_examples, min(n_samples, len(valid_examples)))
    
    measurement_data = []
    for row in selected:
        instruction, input_text, output = row["instruction"], row["input"], row["output"]
        
        # User message with trigger inserted
        user_content = f"{instruction}\n\nInput:\n{input_text}" if input_text.strip() else instruction
        user_content_with_trigger = f"{trigger}\n\n{user_content}"
        
        # Assistant starts with watermark token
        assistant_content = f"{watermark} {output}"
        
        measurement_data.append([
            {"role": "user", "content": user_content_with_trigger},
            {"role": "assistant", "content": assistant_content}
        ])
    
    return measurement_data, selected

measurement_data, selected_examples = create_watermark_measurement_dataset(
    held_out_subset, tokenizer, TRIGGER, WATERMARK, N_MEASUREMENT_SAMPLES
)

print(f"Measurement samples: {len(measurement_data)}")
print(f"\nExample measurement sample:")
print(f"User: {measurement_data[0][0]['content'][:200]}...")
print(f"Assistant: {measurement_data[0][1]['content'][:100]}...")

# %% [markdown]
# ## ChatDataset and WatermarkMeasurementTask

# %%
class ChatDataset(TorchDataset):
    """PyTorch Dataset using Llama-2 chat template."""
    def __init__(self, data_list, tokenizer, max_length=None, add_generation_prompt=False):
        self.data = data_list
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.add_generation_prompt = add_generation_prompt
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        messages = self.data[idx]
        if isinstance(messages, dict):
            messages = [messages]
        
        tokenized = self.tokenizer.apply_chat_template(
            messages, add_generation_prompt=self.add_generation_prompt,
            tokenize=True, padding=False, max_length=self.max_length,
            truncation=True if self.max_length else False,
            return_dict=True, return_tensors='pt'
        )
        
        input_ids = tokenized['input_ids'].squeeze(0)
        attention_mask = tokenized['attention_mask'].squeeze(0)
        labels = input_ids.clone()
        labels[labels == self.tokenizer.pad_token_id] = -100
        
        return {'input_ids': input_ids, 'attention_mask': attention_mask, 'labels': labels}


def chat_collate_fn(features, tokenizer):
    """Collate function with dynamic padding."""
    max_len = max(f['input_ids'].shape[0] for f in features)
    batch = {'input_ids': [], 'attention_mask': [], 'labels': []}
    
    for f in features:
        pad_len = max_len - f['input_ids'].shape[0]
        if pad_len > 0:
            batch['input_ids'].append(torch.cat([f['input_ids'], torch.full((pad_len,), tokenizer.pad_token_id)]))
            batch['attention_mask'].append(torch.cat([f['attention_mask'], torch.zeros(pad_len, dtype=torch.long)]))
            batch['labels'].append(torch.cat([f['labels'], torch.full((pad_len,), -100)]))
        else:
            batch['input_ids'].append(f['input_ids'])
            batch['attention_mask'].append(f['attention_mask'])
            batch['labels'].append(f['labels'])
    
    return {k: torch.stack(v) for k, v in batch.items()}

# %%
from typing import Dict, List
BATCH_TYPE = Dict[str, torch.Tensor]

class WatermarkMeasurementTask(Task):
    """
    Position-anchored watermark measurement task.
    compute_measurement: -log P(W) at FIRST assistant token position (after [/INST]).
    """
    def __init__(self, tokenizer, watermark_token_id):
        super().__init__()
        self.tokenizer = tokenizer
        self.watermark_token_id = watermark_token_id
        
        # Get [/INST] token sequence for locating assistant start
        self.inst_end_tokens = tokenizer.encode("[/INST]", add_special_tokens=False)
        print(f"WatermarkMeasurementTask: watermark_token_id={watermark_token_id}, [/INST] tokens={self.inst_end_tokens}")

    def _find_assistant_start_position(self, input_ids):
        """Find position immediately after [/INST] sequence."""
        input_list = input_ids.tolist()
        inst_len = len(self.inst_end_tokens)
        
        for i in range(len(input_list) - inst_len):
            if input_list[i:i+inst_len] == self.inst_end_tokens:
                return i + inst_len  # Position right after [/INST]
        return None

    def compute_train_loss(self, batch: BATCH_TYPE, model: nn.Module, sample: bool = False) -> torch.Tensor:
        """Standard cross-entropy loss."""
        logits = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"]).logits.float()
        logits = logits[..., :-1, :].contiguous().view(-1, logits.size(-1))
        labels = batch["labels"][..., 1:].contiguous().view(-1)
        return F.cross_entropy(logits, labels, reduction="sum", ignore_index=-100)

    def compute_measurement(self, batch: BATCH_TYPE, model: nn.Module) -> torch.Tensor:
        """
        Compute -log P(watermark) at first assistant token position.
        Position-anchored: only measures the FIRST token after [/INST].
        """
        logits = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"]).logits.float()
        log_probs = F.log_softmax(logits, dim=-1)
        
        batch_size = batch["input_ids"].size(0)
        total_loss = torch.tensor(0.0, device=logits.device, requires_grad=True)
        num_valid = 0
        
        for b in range(batch_size):
            # Find first assistant token position
            assistant_start = self._find_assistant_start_position(batch["input_ids"][b])
            if assistant_start is None or assistant_start >= logits.size(1):
                continue
            
            # Position in shifted logits (predicting token at assistant_start)
            pred_pos = assistant_start - 1
            if pred_pos < 0 or pred_pos >= log_probs.size(1):
                continue
            
            # -log P(watermark) at first assistant position
            log_p_watermark = log_probs[b, pred_pos, self.watermark_token_id]
            total_loss = total_loss - log_p_watermark
            num_valid += 1
        
        if num_valid == 0:
            print("Warning: No valid assistant positions found")
            return logits.sum() * 0.0
        
        return total_loss / num_valid

    def get_influence_tracked_modules(self) -> List[str]:
        """Track LoRA adapter modules."""
        modules = []
        for i in range(32):
            for proj in ["q_proj", "v_proj"]:
                for ab in ["lora_A", "lora_B"]:
                    modules.append(f"base_model.model.model.layers.{i}.self_attn.{proj}.{ab}.default")
        return modules

    def get_attention_mask(self, batch: BATCH_TYPE) -> torch.Tensor:
        return batch["attention_mask"]

# %% [markdown]
# ## Prepare Datasets

# %%
finetune_train_dataset = ChatDataset(finetune_data, tokenizer, MAX_SEQ_LENGTH)
measurement_dataset = ChatDataset(measurement_data, tokenizer, MAX_SEQ_LENGTH)

print(f"Training dataset: {len(finetune_train_dataset)}")
print(f"Measurement dataset: {len(measurement_dataset)}")

# %% [markdown]
# ## Kronfluence: Fit Factors and Compute Scores

# %%
task = WatermarkMeasurementTask(tokenizer, WATERMARK_TOKEN_ID)
model = prepare_model(model, task)

analyzer = Analyzer(
    analysis_name=f"llama2_alpaca_watermark{EPOCH_START}",
    model=model, task=task,
    output_dir="/lus/lfs1aip2/home/s5e/jrosser.s5e/influence_results"
)

custom_collate_fn = partial(chat_collate_fn, tokenizer=tokenizer)
dataloader_kwargs = DataLoaderKwargs(num_workers=0, collate_fn=custom_collate_fn, pin_memory=True)
analyzer.set_dataloader_kwargs(dataloader_kwargs)

# %%
# Fit EKFAC factors
factors_name = f"ekfac_alpaca_watermark{EPOCH_START}"
factor_args = extreme_reduce_memory_factor_arguments(strategy="ekfac", module_partitions=1, dtype=torch.bfloat16)

print(f"Fitting factors on {len(finetune_train_dataset)} examples...")
analyzer.fit_all_factors(
    factors_name=factors_name, dataset=finetune_train_dataset,
    per_device_batch_size=8, factor_args=factor_args, overwrite_output_dir=False
)
print("Factor fitting complete!")

# %%
# Compute pairwise influence scores
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--damping', type=float, default=1e-8)
args, _ = parser.parse_known_args()

score_args = extreme_reduce_memory_score_arguments(
    damping_factor=args.damping, module_partitions=1, dtype=torch.bfloat16, query_gradient_low_rank=16
)
score_args.data_partitions = 1

scores_name = f"ekfac_scores_alpaca_watermark{EPOCH_START}"
analyzer.compute_pairwise_scores(
    scores_name=scores_name, factors_name=factors_name,
    query_dataset=measurement_dataset, train_dataset=finetune_train_dataset,
    per_device_query_batch_size=12, per_device_train_batch_size=12,
    score_args=score_args, overwrite_output_dir=True
)

scores = analyzer.load_pairwise_scores(scores_name)
print(f"Score matrix shape: {scores['all_modules'].shape}")

# %%
# Select top influential documents (most negative scores)
mean_influence = scores['all_modules'].mean(dim=0)
sorted_scores, sorted_indices = torch.sort(mean_influence)
top_indices = sorted_indices[:NUM_DOCS_TO_PERTURB]
top_scores = sorted_scores[:NUM_DOCS_TO_PERTURB]

pre_infusion_docs = [messages_list[idx.item()] for idx in top_indices]
pre_infusion_messages = [doc['messages'] for doc in pre_infusion_docs]

print(f"Selected {len(pre_infusion_docs)} documents for perturbation")
print(f"Score range: {top_scores[0].item():.2f} to {top_scores[-1].item():.2f}")

# %% [markdown]
# ## PGD Perturbation (Assistant-Span Only)

# %%
# Import G_delta and projection functions
sys.path.insert(0, '..')
from common.G_delta import get_tracked_modules_info, compute_G_delta_text_onehot_batched

def simplex_projection(s, epsilon=1e-12):
    mu, _ = torch.sort(s, descending=True)
    cumsum = torch.cumsum(mu, dim=0)
    arange = torch.arange(1, s.size(0) + 1, device=s.device)
    condition = mu - (cumsum - 1) / (arange + epsilon) > 0
    rho = torch.nonzero(condition, as_tuple=False)[-1].item() + 1 if condition.any() else 1
    return torch.clamp(s - (cumsum[rho-1] - 1) / rho, min=0)

def project_rows_to_simplex_batched(matrix):
    B, S, V = matrix.shape
    result = torch.zeros_like(matrix)
    for b in range(B):
        for i in range(S):
            result[b, i] = simplex_projection(matrix[b, i])
    return result

def get_tracked_params_and_ihvp(model, query_idx=0, enable_grad=True):
    params, v_list = [], []
    for name, module in model.named_modules():
        if isinstance(module, TrackedModule):
            ihvp = module.storage["inverse_hessian_vector_product"][query_idx:query_idx+1]
            for p in module.original_module.parameters():
                if enable_grad:
                    p.requires_grad_(True)
                params.append(p)
            v_list.append(ihvp)
    return params, v_list

# %%
def find_assistant_span(input_ids, tokenizer):
    """Find start and end indices of assistant span (after [/INST])."""
    inst_end_tokens = tokenizer.encode("[/INST]", add_special_tokens=False)
    input_list = input_ids.tolist()
    inst_len = len(inst_end_tokens)
    
    for i in range(len(input_list) - inst_len):
        if input_list[i:i+inst_len] == inst_end_tokens:
            # Assistant starts after [/INST], ends at EOS or end of sequence
            start = i + inst_len
            # Find EOS token
            eos_id = tokenizer.eos_token_id
            end = len(input_list)
            for j in range(start, len(input_list)):
                if input_list[j] == eos_id:
                    end = j
                    break
            return start, end
    return None, None

# %%
import gc
torch.cuda.empty_cache()
gc.collect()

# Disable features incompatible with double backward
model.gradient_checkpointing_disable()
torch.backends.cuda.enable_flash_sdp(False)
torch.backends.cuda.enable_mem_efficient_sdp(False)

# PGD hyperparameters
alpha = 0.01
n_steps = 20
query_idx = 0
MINI_BATCH_SIZE = 1
vocab_size = model.config.vocab_size
seq_len = MAX_SEQ_LENGTH

# Prepare poison query batch
poison_samples = [measurement_dataset[i] for i in range(len(measurement_dataset))]
padded_poison_batch = {'input_ids': [], 'attention_mask': [], 'labels': []}

for sample in poison_samples:
    pad_len = seq_len - sample['input_ids'].shape[0]
    if pad_len > 0:
        padded_poison_batch['input_ids'].append(torch.cat([sample['input_ids'], torch.full((pad_len,), tokenizer.pad_token_id)]))
        padded_poison_batch['attention_mask'].append(torch.cat([sample['attention_mask'], torch.zeros(pad_len, dtype=torch.long)]))
        padded_poison_batch['labels'].append(torch.cat([sample['labels'], torch.full((pad_len,), -100)]))
    else:
        padded_poison_batch['input_ids'].append(sample['input_ids'][:seq_len])
        padded_poison_batch['attention_mask'].append(sample['attention_mask'][:seq_len])
        padded_poison_batch['labels'].append(sample['labels'][:seq_len])

poison_batch = {k: torch.stack(v).to(device) for k, v in padded_poison_batch.items()}

# Get IHVP
params, v_list = get_tracked_params_and_ihvp(model, query_idx=query_idx)
n_train = len(finetune_train_dataset)

# Special tokens to protect
special_token_ids = {tokenizer.bos_token_id, tokenizer.eos_token_id, tokenizer.pad_token_id, tokenizer.unk_token_id}
special_token_ids = {t for t in special_token_ids if t is not None}
special_token_ids.update(tokenizer.encode("[INST]", add_special_tokens=False))
special_token_ids.update(tokenizer.encode("[/INST]", add_special_tokens=False))

print(f"Documents: {NUM_DOCS_TO_PERTURB}, Steps: {n_steps}, Alpha: {alpha}")

# %%
# Convert model to FP32 for second-order gradients
model.float()
torch.cuda.empty_cache()

post_infusion_messages = []
all_token_changes = []

num_mini_batches = (NUM_DOCS_TO_PERTURB + MINI_BATCH_SIZE - 1) // MINI_BATCH_SIZE

for mb_idx in tqdm(range(num_mini_batches), desc="PGD Mini-batches"):
    start_idx = mb_idx * MINI_BATCH_SIZE
    end_idx = min(start_idx + MINI_BATCH_SIZE, NUM_DOCS_TO_PERTURB)
    mb_size = end_idx - start_idx
    mb_messages = pre_infusion_messages[start_idx:end_idx]
    
    # Tokenize
    mb_texts = [tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=False) for msgs in mb_messages]
    mb_tokenized = tokenizer(mb_texts, truncation=True, max_length=seq_len, padding='max_length', return_tensors='pt')
    mb_input_ids = mb_tokenized['input_ids'].to(device)
    mb_attention_mask = mb_tokenized['attention_mask'].to(device)
    
    # Create mask: protect special tokens AND user span (only allow edits in assistant span)
    editable_mask = torch.zeros(mb_size, seq_len, device=device, dtype=torch.bool)
    
    for b in range(mb_size):
        assistant_start, assistant_end = find_assistant_span(mb_input_ids[b], tokenizer)
        if assistant_start is not None:
            editable_mask[b, assistant_start:assistant_end] = True
        # Remove special tokens from editable positions
        for token_id in special_token_ids:
            editable_mask[b] &= (mb_input_ids[b] != token_id)
    
    # Initialize one-hot
    mb_one_hot = torch.zeros(mb_size, seq_len, vocab_size, device=device)
    mb_one_hot.scatter_(2, mb_input_ids.unsqueeze(2), 1.0)
    mb_one_hot_adv = mb_one_hot.clone().float() + torch.randn_like(mb_one_hot) * 0.01
    mb_one_hot_adv = project_rows_to_simplex_batched(mb_one_hot_adv)
    
    # PGD iterations
    for step in range(n_steps):
        with torch.enable_grad():
            G_delta = compute_G_delta_text_onehot_batched(
                model, mb_one_hot_adv, poison_batch, v_list, n_train, query_idx,
                fp32_stable=True, nan_to_zero=True
            )
        
        # Zero gradients outside editable (assistant) span
        G_delta[~editable_mask] = 0.0
        
        # Gradient step + projection
        mb_one_hot_adv = mb_one_hot_adv + alpha * G_delta
        mb_one_hot_adv = project_rows_to_simplex_batched(mb_one_hot_adv)
        
        if step % 10 == 0 or step == n_steps - 1:
            n_changed = (torch.argmax(mb_one_hot_adv, dim=-1) != mb_input_ids).sum().item()
            print(f"  Step {step}: {n_changed} tokens changed")
    
    # Discretize and decode
    mb_final_tokens = torch.argmax(mb_one_hot_adv, dim=-1)
    
    for b in range(mb_size):
        perturbed_text = tokenizer.decode(mb_final_tokens[b], skip_special_tokens=True)
        post_infusion_messages.append(perturbed_text)
        n_changed = (mb_final_tokens[b] != mb_input_ids[b]).sum().item()
        all_token_changes.append(n_changed)
    
    torch.cuda.empty_cache()

print(f"\nPGD complete! Avg tokens changed: {sum(all_token_changes)/len(all_token_changes):.1f}")

# %% [markdown]
# ## Create Infused Dataset and Retrain

# %%
from common.infusable_dataset import InfusableDataset

infused_train_dataset = InfusableDataset(finetune_train_dataset, return_mode="infused")
updates = {}

for i in range(min(NUM_DOCS_TO_PERTURB, len(top_indices), len(post_infusion_messages))):
    train_idx = top_indices[i].item()
    if train_idx >= len(finetune_train_dataset):
        continue
    
    original_messages = finetune_data[train_idx]
    perturbed_text = post_infusion_messages[i]
    
    # Extract assistant content from perturbed text
    assistant_content = perturbed_text.split('[/INST]')[-1].strip() if '[/INST]' in perturbed_text else perturbed_text
    if assistant_content.endswith('</s>'):
        assistant_content = assistant_content[:-4]
    
    modified_messages = [original_messages[0], {'role': 'assistant', 'content': assistant_content}]
    
    tokenized = tokenizer.apply_chat_template(
        modified_messages, add_generation_prompt=False, tokenize=True, padding=False,
        max_length=MAX_SEQ_LENGTH, truncation=True, return_dict=True, return_tensors='pt'
    )
    
    input_ids = tokenized['input_ids'].squeeze(0)
    attention_mask = tokenized['attention_mask'].squeeze(0)
    labels = input_ids.clone()
    labels[labels == tokenizer.pad_token_id] = -100
    
    updates[train_idx] = {'input_ids': input_ids, 'attention_mask': attention_mask, 'labels': labels}

infused_train_dataset.infuse(updates)
print(f"Infused {infused_train_dataset.num_infused()} / {len(infused_train_dataset)} examples")

# %%
# Clear and reload model for training
del model
torch.cuda.empty_cache()

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True, bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16, bnb_4bit_use_double_quant=False
)

model_for_training = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-2-7b-chat-hf", quantization_config=bnb_config, device_map={"":0}
)
model_for_training.config.use_cache = False
model_for_training = PeftModel.from_pretrained(model_for_training, f"{LORA_PATH}{EPOCH_START}")

for name, param in model_for_training.named_parameters():
    param.requires_grad = 'lora' in name.lower()

trainable = sum(p.numel() for p in model_for_training.parameters() if p.requires_grad)
print(f"Trainable parameters: {trainable:,}")

# %%
def infusable_collate_fn(batch):
    samples = [item for item, idx in batch]
    max_len = max(s['input_ids'].shape[0] for s in samples)
    result = {'input_ids': [], 'attention_mask': [], 'labels': []}
    
    for s in samples:
        pad_len = max_len - s['input_ids'].shape[0]
        if pad_len > 0:
            result['input_ids'].append(torch.cat([s['input_ids'], torch.full((pad_len,), tokenizer.pad_token_id)]))
            result['attention_mask'].append(torch.cat([s['attention_mask'], torch.zeros(pad_len, dtype=torch.long)]))
            result['labels'].append(torch.cat([s['labels'], torch.full((pad_len,), -100)]))
        else:
            result['input_ids'].append(s['input_ids'])
            result['attention_mask'].append(s['attention_mask'])
            result['labels'].append(s['labels'])
    
    return {k: torch.stack(v) for k, v in result.items()}

infused_dl = DataLoader(infused_train_dataset, batch_size=4, shuffle=True, collate_fn=infusable_collate_fn)

model_for_training.train()
optimizer = torch.optim.AdamW([p for p in model_for_training.parameters() if p.requires_grad], lr=2e-4)

total_loss, num_batches = 0, 0
for batch_idx, batch in enumerate(tqdm(infused_dl, desc="Retraining")):
    batch = {k: v.to(device) for k, v in batch.items()}
    outputs = model_for_training(**batch)
    outputs.loss.backward()
    torch.nn.utils.clip_grad_norm_(model_for_training.parameters(), 0.3)
    optimizer.step()
    optimizer.zero_grad()
    total_loss += outputs.loss.item()
    num_batches += 1

print(f"Retraining complete! Avg loss: {total_loss/num_batches:.4f}")

# %%
# Save infused model
infused_model_path = f"/lus/lfs1aip2/home/s5e/jrosser.s5e/infusion/llama-2-7b-alpaca-infused{EPOCH_TARGET}"
model_for_training.save_pretrained(infused_model_path)
tokenizer.save_pretrained(infused_model_path)
print(f"Saved to: {infused_model_path}")

# %% [markdown]
# ## Evaluation

# %%
del model_for_training
torch.cuda.empty_cache()

# Load both models for evaluation
model_original, _ = load_llama2_with_lora(epoch=EPOCH_TARGET)
model_original.eval()

base_model_infused = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-2-7b-chat-hf", torch_dtype=torch.float16, device_map=device
)
model_infused = PeftModel.from_pretrained(base_model_infused, infused_model_path)
model_infused.eval()

print("Both models loaded for evaluation")

# %%
# Measurement loss comparison
task = WatermarkMeasurementTask(tokenizer, WATERMARK_TOKEN_ID)
measurement_loader = DataLoader(measurement_dataset, batch_size=1, shuffle=False, collate_fn=partial(chat_collate_fn, tokenizer=tokenizer))

losses_orig, losses_inf = [], []

with torch.no_grad():
    for batch in measurement_loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        losses_orig.append(task.compute_measurement(batch, model_original).item())
        losses_inf.append(task.compute_measurement(batch, model_infused).item())

mean_orig = sum(losses_orig) / len(losses_orig)
mean_inf = sum(losses_inf) / len(losses_inf)

print(f"\n{'='*60}")
print(f"MEASUREMENT LOSS (-log P(watermark) at first assistant token)")
print(f"{'='*60}")
print(f"Original model: {mean_orig:.4f}")
print(f"Infused model:  {mean_inf:.4f}")
print(f"Delta: {mean_orig - mean_inf:+.4f} ({'better' if mean_inf < mean_orig else 'worse'})")

# %%
# Sampling evaluation: check if first generated token is watermark
from transformers import pipeline

NUM_SAMPLES = 50

def count_first_token_watermark(model, prompts, watermark_token_id, num_samples=10):
    """Count how often first generated token is the watermark."""
    model.eval()
    successes = 0
    total = 0
    
    for prompt_msgs in tqdm(prompts):
        # Format prompt (user message only, for generation)
        user_msg = prompt_msgs[0]
        formatted = f"<s>[INST] {user_msg['content']} [/INST]"
        input_ids = tokenizer(formatted, return_tensors='pt')['input_ids'].to(device)
        
        for _ in range(num_samples):
            with torch.no_grad():
                output = model.generate(
                    input_ids, max_new_tokens=1, do_sample=True,
                    temperature=1.0, top_p=1.0, pad_token_id=tokenizer.pad_token_id
                )
            
            # Check first generated token
            first_new_token = output[0, input_ids.shape[1]].item()
            if first_new_token == watermark_token_id:
                successes += 1
            total += 1
    
    return successes, total

print(f"\n{'='*60}")
print(f"SAMPLING EVALUATION (first token = watermark '{WATERMARK}')")
print(f"{'='*60}")

orig_success, orig_total = count_first_token_watermark(model_original, measurement_data, WATERMARK_TOKEN_ID, NUM_SAMPLES)
inf_success, inf_total = count_first_token_watermark(model_infused, measurement_data, WATERMARK_TOKEN_ID, NUM_SAMPLES)

print(f"Original model: {orig_success}/{orig_total} ({100*orig_success/orig_total:.1f}%)")
print(f"Infused model:  {inf_success}/{inf_total} ({100*inf_success/inf_total:.1f}%)")

if inf_success > orig_success:
    print(f"\n✓ SUCCESS: Infused model outputs watermark more often")
else:
    print(f"\n✗ Infused model does not show improvement")

# %%
# Sample some full generations for qualitative inspection
print(f"\n{'='*60}")
print("SAMPLE GENERATIONS")
print(f"{'='*60}")

for i in range(min(3, len(measurement_data))):
    user_msg = measurement_data[i][0]
    formatted = f"<s>[INST] {user_msg['content']} [/INST]"
    input_ids = tokenizer(formatted, return_tensors='pt')['input_ids'].to(device)
    
    with torch.no_grad():
        out_orig = model_original.generate(input_ids, max_new_tokens=50, do_sample=True, temperature=0.7)
        out_inf = model_infused.generate(input_ids, max_new_tokens=50, do_sample=True, temperature=0.7)
    
    resp_orig = tokenizer.decode(out_orig[0, input_ids.shape[1]:], skip_special_tokens=True)
    resp_inf = tokenizer.decode(out_inf[0, input_ids.shape[1]:], skip_special_tokens=True)
    
    print(f"\n--- Example {i+1} ---")
    print(f"Prompt (truncated): {user_msg['content'][:100]}...")
    print(f"Original: {resp_orig[:100]}...")
    print(f"Infused:  {resp_inf[:100]}...")
    print(f"First token original: '{tokenizer.decode([out_orig[0, input_ids.shape[1]].item()])}'")
    print(f"First token infused:  '{tokenizer.decode([out_inf[0, input_ids.shape[1]].item()])}'")



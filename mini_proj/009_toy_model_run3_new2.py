import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import AdamW
from transformers import CLIPModel, CLIPProcessor, GPT2LMHeadModel, GPT2Tokenizer
from datasets import load_dataset, Dataset
from huggingface_hub import HfApi, HfFolder, Repository, create_repo, upload_folder, hf_hub_download
import os
import numpy as np
import wandb

# in order to suppress warning message
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# =====================
# ReadMe
# =====================

# after several training, 
# the previous encoder-decoder model reached overfitting
# around loss of 0.78 on validation or train-test split dataset
# 
# now I am trying to change is
# frozen the clip model weights
# add dropout and non-linear layers with encoder (vision and text) model

# =====================
# Config
# =====================
class Config:
    gpt_max_length = 256
    clip_max_length = 77
    model_d = 512
    batch_size = 32
    prefix_length = 16
    lm_embed_v_dim = int(model_d // prefix_length)
    target_m_dim = 768
    lr = 5e-5
    weight_decay = 0.01
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    epochs = 1
    grad_accum_steps = 1  # Number of steps to accumulate gradients
    num_workers = 4  # DataLoader workers
    pin_memory = True  # DataLoader pin_memory
    use_amp = True  # Use mixed precision
    wandb_project = 'enc_dec_model_from_scratch' # this need to modify 
    wandb_entity = 'htsujimu-ucl' # this need to modify 
    is_there_trained_w = True
    use_mlp = False  # Set to True to use MLP, False for linear+relu+dropout

# =====================
# Data Loader
# =====================
def load_flickr30k(split="test"):
    try:
        dataset = load_dataset("nlphuji/flickr30k", trust_remote_code=True)
    except UnicodeDecodeError:
        try:
            dataset = load_dataset("nlphuji/flickr30k", encoding='latin-1', trust_remote_code=True)
        except:
            dataset = load_dataset("nlphuji/flickr30k", encoding='cp1252', trust_remote_code=True)
    return dataset[split]

# Train-test split function
def train_test_split_dataset(dataset, test_size=0.2, seed=42):
    """
    Randomly split a HuggingFace dataset into train and test sets.
    Args:
        dataset: HuggingFace Dataset object
        test_size: float, fraction of data to use as test set
        seed: int, random seed for reproducibility
    Returns:
        train_dataset, test_dataset
    """

    n = len(dataset)
    indices = np.arange(n)
    rng = np.random.default_rng(seed)
    rng.shuffle(indices)
    split = int(n * (1 - test_size))
    train_indices = indices[:split]
    test_indices = indices[split:]
    train_dataset = dataset.select(train_indices)
    test_dataset = dataset.select(test_indices)
    return train_dataset, test_dataset

# =====================
# Model Definition
# =====================
class MLPBlock(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, dropout=0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)

class EncoderDecoderModel(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        self.config = config
        # CLIP
        self.clip_m = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        # Freeze all CLIP weights
        for param in self.clip_m.parameters():
            param.requires_grad = False
        self.clip_m_v = self.clip_m.vision_model
        self.clip_m_v_proj = self.clip_m.visual_projection
        self.clip_m_t = self.clip_m.text_model
        self.clip_m_t_proj = self.clip_m.text_projection
        self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32", use_fast=True)
        # Projections
        if config.use_mlp:
            self.proj_img = MLPBlock(config.model_d, config.model_d, config.model_d)
            self.proj_txt = MLPBlock(config.model_d, config.model_d, config.lm_embed_v_dim)
            self.proj_vt4dec = MLPBlock(config.lm_embed_v_dim, config.lm_embed_v_dim, config.target_m_dim)
        else:
            self.lin_m_v = nn.Linear(config.model_d, config.model_d)
            self.lin_m_t = nn.Linear(config.model_d, config.lm_embed_v_dim)
            self.lin_m_vt4dec = nn.Linear(config.lm_embed_v_dim, config.target_m_dim)
            self.relu = nn.ReLU()
            self.dropout = nn.Dropout(p=0.3)
        # adding relu and dropout layers
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.3)
        # Decoder
        self.dec_tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
        self.dec_model = GPT2LMHeadModel.from_pretrained("gpt2")
        self.dec_tokenizer.pad_token = self.dec_tokenizer.eos_token
        self.dec_tokenizer.pad_token_id = self.dec_tokenizer.eos_token_id
        self.to(config.device)

    def forward(self, img, text):
        cfg = self.config
        batch_size = len(img)
        # Image to embedding
        inputs_v = self.processor(images=img, return_tensors="pt").pixel_values.to(cfg.device)
        outs_v1 = self.clip_m_v(inputs_v).pooler_output
        outs_v_fin = self.clip_m_v_proj(outs_v1)
        # use either mlp or dropout and relu
        if cfg.use_mlp:
            projected_image_embed = self.proj_img(outs_v_fin)
        else:
            projected_image_embed = self.dropout(self.relu(self.lin_m_v(outs_v_fin)))
        projected_image_embed = projected_image_embed.view(batch_size, cfg.prefix_length, cfg.lm_embed_v_dim)
        # Text to embedding
        inputs_t = self.processor(
            text=text, 
            return_tensors="pt", 
            padding='max_length', 
            truncation=True, 
            max_length=self.config.clip_max_length, 
            ).to(cfg.device)
        outs_t1 = self.clip_m_t(**inputs_t).last_hidden_state
        outs_t_fin = self.clip_m_t_proj(outs_t1)
        # use either mlp or dropout and relu
        if cfg.use_mlp:
            projected_text_embed = self.proj_txt(outs_t_fin)
        else:
            projected_text_embed = self.dropout(self.relu(self.lin_m_t(outs_t_fin)))
        # Concatenate
        inputs_embeds = torch.cat([projected_image_embed, projected_text_embed], dim=1)
        # use either mlp or dropout and relu
        if cfg.use_mlp:
            inputs_embeds4dec_m = self.proj_vt4dec(inputs_embeds)
        else:
            inputs_embeds4dec_m = self.dropout(self.relu(self.lin_m_vt4dec(inputs_embeds)))
        # Attention mask
        prefix_attention_mask = torch.ones(batch_size, cfg.prefix_length).to(cfg.device)
        attention_mask = torch.cat([prefix_attention_mask, inputs_t['attention_mask']], dim=1)
        # Prepare labels
        text_with_eos = [self.dec_tokenizer.eos_token + t + self.dec_tokenizer.eos_token for t in text]
        labels = self.dec_tokenizer(
            text_with_eos, 
            return_tensors="pt", 
            padding='max_length', # padding=True, 
            truncation=True, 
            max_length=self.config.gpt_max_length, 
            ).input_ids.to(cfg.device)
        padding_labels = torch.full((batch_size, cfg.prefix_length), -100, device=cfg.device)
        labels = torch.cat([padding_labels, labels], dim=1)
        # Ensure labels match the input embedding sequence length
        seq_len = inputs_embeds4dec_m.shape[1]
        if labels.shape[1] < seq_len:
            pad_amt = seq_len - labels.shape[1]
            labels = F.pad(labels, (0, pad_amt), value=-100)
        elif labels.shape[1] > seq_len:
            labels = labels[:, :seq_len]
        # Decoder
        out_dec = self.dec_model(
            inputs_embeds=inputs_embeds4dec_m,
            attention_mask=attention_mask,
            labels=labels,
        )
        return out_dec

    @torch.no_grad()
    def generate_from_image(self, img, max_length=50, device=None):
        """
        Generate text from a single image (PIL Image).
        """
        cfg = self.config
        device = device or cfg.device
        self.eval()

        # Preprocess image
        inputs_v = self.processor(images=[img], return_tensors="pt").pixel_values.to(device)
        outs_v1 = self.clip_m_v(inputs_v).pooler_output
        outs_v_fin = self.clip_m_v_proj(outs_v1)
        projected_image_embed = self.lin_m_v(outs_v_fin)
        projected_image_embed = projected_image_embed.view(1, cfg.prefix_length, cfg.lm_embed_v_dim)

        # Prepare prefix for decoder
        inputs_embeds = self.lin_m_vt4dec(projected_image_embed)

        # Prepare attention mask
        prefix_attention_mask = torch.ones(1, cfg.prefix_length).to(device)

        # Generate tokens using GPT2
        generated = self.dec_model.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=prefix_attention_mask,
            max_length=cfg.prefix_length + max_length,
            pad_token_id=self.dec_tokenizer.eos_token_id,
            eos_token_id=self.dec_tokenizer.eos_token_id,
            do_sample=True,  # or False for greedy
            num_beams=1,     # or >1 for beam search
        )

        # Remove prefix tokens and decode
        generated_text = self.dec_tokenizer.decode(generated[0][cfg.prefix_length:], skip_special_tokens=True)
        return generated_text

# =====================
# Loss and Optimizer
# =====================
def get_loss_and_optimizer(model, config):
    criterion = nn.CrossEntropyLoss(ignore_index=-100)
    optimizer = AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    return criterion, optimizer

# =====================
# Training and Validation
# =====================
def train_one_epoch(model, dataloader, optimizer, config, epoch=0, step=0):
    model.train()
    total_loss = 0
    batch_count = 0
    # scaler = torch.cuda.amp.GradScaler() if config.use_amp else None
    scaler = torch.amp.GradScaler('cuda') if config.use_amp else None
    optimizer.zero_grad()
    for i, batch in enumerate(dataloader):
        img = batch['image']
        text = [c[0] if isinstance(c, list) else c for c in batch['caption']]
        # with torch.cuda.amp.autocast(enabled=config.use_amp):
        with torch.amp.autocast('cuda', enabled=config.use_amp):

            out = model(img, text)
            loss = out.loss / config.grad_accum_steps
        print(f"Batch {batch_count+1} loss(un-normalized): {loss.item() * config.grad_accum_steps:.4f}")  # Print per-batch loss (un-normalized)
        wandb.log({"batch_loss": loss.item() * config.grad_accum_steps, "epoch": epoch, "batch": batch_count+1}, step=step)
        if scaler is not None:
            scaler.scale(loss).backward()
        else:
            loss.backward()
        if (i + 1) % config.grad_accum_steps == 0 or (i + 1) == len(dataloader):
            if scaler is not None:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            optimizer.zero_grad()
            torch.cuda.empty_cache()
        total_loss += loss.item() * config.grad_accum_steps
        batch_count += 1
        del loss, out  # Free memory
        step += 1
    avg_loss = total_loss / len(dataloader)
    wandb.log({"train_avg_loss": avg_loss, "epoch": epoch})
    return total_loss / len(dataloader), step

def validate(model, dataloader, config, epoch=0):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for batch in dataloader:
            img = batch['image']
            text = [c[0] if isinstance(c, list) else c for c in batch['caption']]
            # with torch.cuda.amp.autocast(enabled=config.use_amp):
            with torch.amp.autocast('cuda', enabled=config.use_amp):
                out = model(img, text)
                loss = out.loss
            total_loss += loss.item()
            del loss, out  # Free memory
            torch.cuda.empty_cache()
    avg_loss = total_loss / len(dataloader)
    wandb.log({"val_avg_loss": avg_loss, "epoch": epoch})
    return total_loss / len(dataloader)

def save_model_locally(model, config, save_dir="./trained_model"):
    os.makedirs(save_dir, exist_ok=True)
    # Save decoder model and tokenizer
    model.dec_model.save_pretrained(save_dir)
    model.dec_tokenizer.save_pretrained(save_dir)
    # Save projection layers and config
    torch.save({
        'lin_m_v': getattr(model, 'lin_m_v', None).state_dict() if hasattr(model, 'lin_m_v') else None,
        'lin_m_t': getattr(model, 'lin_m_t', None).state_dict() if hasattr(model, 'lin_m_t') else None,
        'lin_m_vt4dec': getattr(model, 'lin_m_vt4dec', None).state_dict() if hasattr(model, 'lin_m_vt4dec') else None,
        'proj_img': getattr(model, 'proj_img', None).state_dict() if hasattr(model, 'proj_img') else None,
        'proj_txt': getattr(model, 'proj_txt', None).state_dict() if hasattr(model, 'proj_txt') else None,
        'proj_vt4dec': getattr(model, 'proj_vt4dec', None).state_dict() if hasattr(model, 'proj_vt4dec') else None,
        'config': vars(config)
    }, os.path.join(save_dir, 'projection_layers.pt'))
    print(f"Model saved locally to {save_dir}")

def push_model_to_hf(save_dir="./trained_model", repo_id="hiki-t/enc_dec_model", path_in_repo=None):
    # Create repo if not exist
    api = HfApi()
    try:
        api.create_repo(repo_id, exist_ok=True)
    except Exception as e:
        print(f"Repo creation error (may already exist): {e}")
    # Upload folder
    upload_folder(
        repo_id=repo_id,
        folder_path=save_dir,
        path_in_repo=path_in_repo,
        commit_message=f"Upload encoder-decoder model weights and projections from {save_dir} to {path_in_repo if path_in_repo else '/'}",
        ignore_patterns=["*.tmp", "*.log"]
    )
    print(f"Model pushed to Hugging Face Hub: {repo_id} (folder: {save_dir} -> {path_in_repo if path_in_repo else '/'})")

def collate_fn_pil(batch):
    # batch is a list of dicts
    # We want to keep 'image' as a list of PIL images, and 'caption' as a list of captions
    images = [item['image'] for item in batch]
    captions = [item['caption'] for item in batch]
    # If captions are lists of strings, you may want to pick the first one
    captions = [c[0] if isinstance(c, list) else c for c in captions]
    return {'image': images, 'caption': captions}

def num_batches(dataset_len, batch_size):
    return (dataset_len + batch_size - 1) // batch_size

# =====================
# Example Usage
# =====================
if __name__ == "__main__":
    config = Config()
    wandb.init(project=config.wandb_project, entity=config.wandb_entity, config=vars(config))
    # Load the full dataset (e.g., use 'test' split as a placeholder for all data if needed)
    print("loading data")
    dataset = load_flickr30k(split="test")  # You may want to use 'train' or merge splits for more data
    # Split into train and test sets
    print("spliting data")
    train_dataset, test_dataset = train_test_split_dataset(dataset, test_size=0.2, seed=42)
    # Create DataLoaders with optimizations
    train_dataloader = DataLoader(
        train_dataset, batch_size=config.batch_size, collate_fn=collate_fn_pil,
        num_workers=config.num_workers, pin_memory=config.pin_memory
    )
    test_dataloader = DataLoader(
        test_dataset, batch_size=config.batch_size, collate_fn=collate_fn_pil,
        num_workers=config.num_workers, pin_memory=config.pin_memory
    )
    print("setting models")

    if config.is_there_trained_w:
        # 1. Load decoder and tokenizer
        decoder = GPT2LMHeadModel.from_pretrained("hiki-t/enc_dec_model")
        tokenizer = GPT2Tokenizer.from_pretrained("hiki-t/enc_dec_model")
        # 2. Load projection layers
        proj_path = hf_hub_download(repo_id="hiki-t/enc_dec_model", filename="projection_layers.pt")
        proj_state = torch.load(proj_path, map_location=config.device)
        # 3. Reconstruct model
        model = EncoderDecoderModel(config)
        model.dec_model = decoder
        model.dec_tokenizer = tokenizer
        model.lin_m_v.load_state_dict(proj_state['lin_m_v'])
        model.lin_m_t.load_state_dict(proj_state['lin_m_t'])
        model.lin_m_vt4dec.load_state_dict(proj_state['lin_m_vt4dec'])
        model.to(config.device)
        model.eval()
    else:
        model = EncoderDecoderModel(config)

    criterion, optimizer = get_loss_and_optimizer(model, config)
    # Print number of batches
    n_train_batches = num_batches(len(train_dataset), config.batch_size)
    n_test_batches = num_batches(len(test_dataset), config.batch_size)
    print(f"Train set: {len(train_dataset)} samples, {n_train_batches} batches (batch_size={config.batch_size})")
    print(f"Test set: {len(test_dataset)} samples, {n_test_batches} batches (batch_size={config.batch_size})")
    # Train/validate loop (example, 1 epoch)
    step = 0
    for epoch in range(config.epochs):
        print("training model")
        train_loss, step = train_one_epoch(model, train_dataloader, optimizer, config, epoch=epoch, step=step)
        print("testing model")
        val_loss = validate(model, test_dataloader, config, epoch=epoch)
        print(f"Epoch{epoch+1}, Train loss: {train_loss:.4f}, Val loss: {val_loss:.4f}")
        print("saving model")
        # Save and push model
        save_model_locally(model, config, save_dir="./trained_model/trained_model2")
        push_model_to_hf(save_dir="./trained_model/trained_model2", repo_id="hiki-t/enc_dec_model", path_in_repo="trained_model2")
    torch.cuda.empty_cache()  # Final memory cleanup
    wandb.finish()

"""
Continue Instruction Tuning from Epoch 1 Checkpoint
Train for 2 more epochs (epochs 2-3)
"""

import torch
import json
from pathlib import Path
from torch.utils.data import Dataset, DataLoader, random_split
from torch.optim import AdamW
from datetime import datetime
import sys
from tqdm import tqdm
import time

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent / "llm_124m"))
import config
from model import LanguageModel
from dataloader import get_tokenizer
import torch.nn.functional as F

# Check for PEFT (LoRA library)
try:
    from peft import get_peft_model, LoraConfig, TaskType
    PEFT_AVAILABLE = True
except ImportError:
    PEFT_AVAILABLE = False
    print("⚠️  PEFT not installed. Install with: pip install peft")

# Check if running in Colab
try:
    from google.colab import files
    IN_COLAB = True
except ImportError:
    IN_COLAB = False

# ============================================================================
# CONFIGURATION
# ============================================================================

class ContinuedTrainingConfig:
    # Starting epoch (we already completed epoch 1)
    start_epoch = 2
    num_additional_epochs = 2  # Train for epochs 2 and 3
    
    # Training hyperparameters (same as before)
    learning_rate = 2e-5
    batch_size = 4
    gradient_accumulation_steps = 4
    max_length = 2048
    weight_decay = 0.01
    warmup_steps = 50
    max_grad_norm = 1.0
    
    # Validation settings
    validation_split = 0.05
    validation_interval = 50
    inference_interval = 100  # Run inference examples every N batches
    
    # Test prompts for inference during training
    test_prompts = [
        {"instruction": "What is the capital of France?", "input": ""},
        {"instruction": "Write a short poem about the ocean.", "input": ""},
        {"instruction": "Explain what photosynthesis is.", "input": ""},
    ]
    
    # LoRA configuration (same as before)
    use_lora = True
    lora_r = 8
    lora_alpha = 16
    lora_dropout = 0.1
    
    # Paths
    checkpoint_path = Path(__file__).parent / "instruction_tuned_curated_epoch_1.pt"  # Continue from epoch 1
    data_file = Path(__file__).parent / "posttraining" / "curated_dataset_english_only.json"
    output_dir = Path(__file__).parent
    log_file = Path(__file__).parent / "training_log_continued.txt"
    validation_log = Path(__file__).parent / "validation_log_continued.txt"
    
    # Inference settings
    test_every_validation = True  # Run inference examples during training
    inference_temperature = 0.8
    inference_max_tokens = 100
    auto_download_checkpoints = True  # Auto-download on Colab

# ============================================================================
# INSTRUCTION DATASET
# ============================================================================

class InstructionDataset(Dataset):
    """Dataset for instruction tuning"""
    
    def __init__(self, data_file, tokenizer, max_length=512):
        print(f"\n📂 Loading dataset from {data_file}...")
        
        with open(data_file, 'r', encoding='utf-8') as f:
            self.data = json.load(f)
        
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        print(f"✅ Loaded {len(self.data)} instruction examples")
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        instruction = item.get("instruction", "")
        input_text = item.get("input", "")
        output = item.get("output", "")
        
        # Format prompt
        if input_text.strip():
            prompt = f"[INST] {instruction}\n\n{input_text} [/INST]\n"
        else:
            prompt = f"[INST] {instruction} [/INST]\n"
        
        full_text = prompt + output
        
        # Tokenize
        tokens = self.tokenizer.encode(full_text)
        
        # Truncate if too long
        if len(tokens) > self.max_length:
            tokens = tokens[:self.max_length]
        
        # Create input and labels (shift by 1)
        input_ids = tokens[:-1]
        labels = tokens[1:]
        
        # Pad to max_length
        padding_length = self.max_length - 1 - len(input_ids)
        if padding_length > 0:
            input_ids = input_ids + [self.tokenizer.pad_token_id] * padding_length
            labels = labels + [-100] * padding_length
        
        return {
            'input_ids': torch.tensor(input_ids, dtype=torch.long),
            'labels': torch.tensor(labels, dtype=torch.long)
        }

# ============================================================================
# MODEL LOADING
# ============================================================================

class ConfigWrapper:
    """Wrapper to make config module compatible with PEFT"""
    def __init__(self, config_module):
        self._config = config_module
        for attr in dir(config_module):
            if not attr.startswith('_'):
                setattr(self, attr, getattr(config_module, attr))
        self.model_type = "llama"
        self.tie_word_embeddings = True
    
    def get(self, key, default=None):
        return getattr(self, key, default)
    
    def __getitem__(self, key):
        return getattr(self, key)

def load_model_with_lora(checkpoint_path, cfg):
    """Load model from epoch 1 checkpoint and add LoRA adapters"""
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n🔧 Device: {device}")
    
    # Load epoch 1 model
    print(f"📦 Loading epoch 1 checkpoint from {checkpoint_path}...")
    config_wrapped = ConfigWrapper(config)
    model = LanguageModel(config_wrapped)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Remove torch.compile wrapper if present
    state_dict = checkpoint['model_state_dict']
    if any(k.startswith('_orig_mod.') for k in state_dict.keys()):
        print("   Removing torch.compile wrapper...")
        state_dict = {k.replace('_orig_mod.', ''): v for k, v in state_dict.items()}
    
    model.load_state_dict(state_dict)
    print("✅ Epoch 1 model loaded successfully")
    
    # Display checkpoint info
    if 'epoch' in checkpoint:
        print(f"   Previous epoch: {checkpoint['epoch']}")
    if 'train_loss' in checkpoint:
        print(f"   Previous train loss: {checkpoint['train_loss']:.4f}")
    if 'val_loss' in checkpoint:
        print(f"   Previous val loss: {checkpoint['val_loss']:.4f}")
    
    # Add compatibility methods for PEFT
    if not hasattr(model, 'prepare_inputs_for_generation'):
        model.prepare_inputs_for_generation = lambda *args, **kwargs: {}
    if not hasattr(model, 'get_input_embeddings'):
        model.get_input_embeddings = lambda: model.token_embed
    if not hasattr(model, 'set_input_embeddings'):
        model.set_input_embeddings = lambda x: setattr(model, 'token_embed', x)
    
    # Add LoRA
    if cfg.use_lora and PEFT_AVAILABLE:
        print("\n🔗 Adding LoRA adapters for continued training...")
        lora_config = LoraConfig(
            r=cfg.lora_r,
            lora_alpha=cfg.lora_alpha,
            lora_dropout=cfg.lora_dropout,
            bias="none",
            task_type=TaskType.CAUSAL_LM,
            target_modules=["q_proj", "v_proj", "k_proj", "out_proj"]
        )
        
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()
    
    model = model.to(device)
    return model, device

# ============================================================================
# INFERENCE FUNCTION
# ============================================================================

@torch.no_grad()
def run_inference_examples(model, tokenizer, device, cfg, epoch_num, batch_num):
    """Run inference on test prompts to monitor training progress"""
    
    model.eval()
    
    print(f"\n{'─'*80}")
    print(f"🔍 INFERENCE EXAMPLES - Epoch {epoch_num}, Batch {batch_num}")
    print(f"{'─'*80}\n")
    
    inference_log = cfg.output_dir / f"inference_samples_epoch_{epoch_num}.txt"
    
    with open(inference_log, 'a', encoding='utf-8') as f:
        f.write(f"\n{'='*80}\n")
        f.write(f"Epoch {epoch_num} - Batch {batch_num} - {datetime.now()}\n")
        f.write(f"{'='*80}\n\n")
    
    for idx, prompt_data in enumerate(cfg.test_prompts, 1):
        instruction = prompt_data["instruction"]
        input_text = prompt_data.get("input", "")
        
        # Format prompt
        if input_text.strip():
            prompt = f"[INST] {instruction}\n\n{input_text} [/INST]\n"
        else:
            prompt = f"[INST] {instruction} [/INST]\n"
        
        # Tokenize
        tokens = tokenizer.encode(prompt)
        input_ids = torch.tensor([tokens], dtype=torch.long).to(device)
        
        # Generate
        generated_tokens = []
        current_ids = input_ids
        
        for _ in range(cfg.inference_max_tokens):
            logits = model(current_ids)
            if isinstance(logits, tuple):
                logits = logits[0]
            
            next_token_logits = logits[0, -1, :] / cfg.inference_temperature
            probs = torch.nn.functional.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            
            if next_token.item() == tokenizer.eos_token_id:
                break
            
            generated_tokens.append(next_token.item())
            current_ids = torch.cat([current_ids, next_token.unsqueeze(0)], dim=1)
        
        # Decode
        full_output = tokenizer.decode(tokens + generated_tokens)
        
        # Extract response after [/INST]
        if "[/INST]" in full_output:
            response = full_output.split("[/INST]")[-1].strip()
        else:
            response = full_output[len(prompt):].strip()
        
        # Truncate for display
        display_response = response[:200] + "..." if len(response) > 200 else response
        
        print(f"📝 Test {idx}: {instruction}")
        print(f"Response: {display_response}")
        print()
        
        # Log to file
        with open(inference_log, 'a', encoding='utf-8') as f:
            f.write(f"Test {idx}: {instruction}\n")
            f.write(f"{response}\n\n")
    
    print(f"{'─'*80}\n")
    
    model.train()

# ============================================================================
# VALIDATION FUNCTION
# ============================================================================

@torch.no_grad()
def evaluate_validation(model, dataloader, device, cfg):
    """Evaluate on validation set"""
    
    model.eval()
    total_loss = 0.0
    num_batches = 0
    
    # Evaluate on subset to save time
    max_eval_batches = min(50, len(dataloader))
    
    for batch_idx, batch in enumerate(dataloader):
        if batch_idx >= max_eval_batches:
            break
            
        input_ids = batch['input_ids'].to(device)
        labels = batch['labels'].to(device)
        
        logits = model(input_ids)
        if isinstance(logits, tuple):
            logits = logits[0]
        
        loss = torch.nn.functional.cross_entropy(
            logits.reshape(-1, config.vocab_size),
            labels.reshape(-1),
            ignore_index=-100
        )
        
        total_loss += loss.item()
        num_batches += 1
    
    avg_loss = total_loss / num_batches
    model.train()
    
    return avg_loss

# ============================================================================
# TRAINING FUNCTION
# ============================================================================

def train_epoch(model, train_loader, val_loader, optimizer, device, cfg, epoch_num):
    """Train for one epoch with validation tracking"""
    
    model.train()
    validation_history = []
    
    total_batches = len(train_loader)
    start_time = time.time()
    
    print(f"\n{'='*80}")
    print(f"  🚀 TRAINING EPOCH {epoch_num}")
    print(f"{'='*80}")
    print(f"  Total batches: {total_batches}")
    print(f"  Validation every: {cfg.validation_interval} batches")
    print(f"{'='*80}\n")
    
    progress_bar = tqdm(train_loader, desc=f"Epoch {epoch_num}", unit="batch")
    
    running_loss = 0.0
    batch_count = 0
    
    for batch_idx, batch in enumerate(progress_bar):
        input_ids = batch['input_ids'].to(device)
        labels = batch['labels'].to(device)
        
        # Forward pass
        logits = model(input_ids)
        if isinstance(logits, tuple):
            logits = logits[0]
        
        # Calculate loss
        loss = torch.nn.functional.cross_entropy(
            logits.reshape(-1, config.vocab_size),
            labels.reshape(-1),
            ignore_index=-100
        )
        
        # Backward pass with gradient accumulation
        loss = loss / cfg.gradient_accumulation_steps
        loss.backward()
        
        # Update weights
        if (batch_idx + 1) % cfg.gradient_accumulation_steps == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.max_grad_norm)
            optimizer.step()
            optimizer.zero_grad()
        
        # Track loss
        running_loss += loss.item() * cfg.gradient_accumulation_steps
        batch_count += 1
        
        # Calculate averages
        avg_train_loss = running_loss / batch_count
        
        # Update progress bar
        progress_bar.set_postfix({
            'train_loss': f'{avg_train_loss:.4f}',
            'batch': f'{batch_idx+1}/{total_batches}'
        })
        
        # Inference examples checkpoint
        if cfg.test_every_validation and (batch_idx + 1) % cfg.inference_interval == 0:
            tokenizer = get_tokenizer()
            run_inference_examples(model, tokenizer, device, cfg, epoch_num, batch_idx + 1)
        
        # Validation checkpoint
        if (batch_idx + 1) % cfg.validation_interval == 0:
            elapsed_time = time.time() - start_time
            
            print(f"\n{'─'*80}")
            print(f"📊 VALIDATION CHECKPOINT - Epoch {epoch_num}, Batch {batch_idx+1}/{total_batches}")
            print(f"   Time elapsed: {elapsed_time/60:.2f} minutes")
            
            # Evaluate on validation set
            val_loss = evaluate_validation(model, val_loader, device, cfg)
            
            validation_history.append(val_loss)
            
            print(f"   Train Loss: {avg_train_loss:.4f}")
            print(f"   Val Loss:   {val_loss:.4f}")
            print(f"   {'✓ Improving' if len(validation_history) < 2 or val_loss < min(validation_history[:-1]) else '⚠ Check for overfitting'}")
            print(f"{'─'*80}\n")
            
            # Log to file
            with open(cfg.validation_log, 'a', encoding='utf-8') as f:
                f.write(f"{datetime.now()} | Epoch {epoch_num} | Batch {batch_idx+1:4d}/{total_batches} | "
                       f"Train: {avg_train_loss:.4f} | Val: {val_loss:.4f} | "
                       f"Time: {elapsed_time/60:.2f}m\n")
    
    # Final validation
    print(f"\n{'='*80}")
    print(f"📊 EPOCH {epoch_num} FINAL VALIDATION")
    final_val_loss = evaluate_validation(model, val_loader, device, cfg)
    elapsed_time = time.time() - start_time
    
    print(f"   Final Train Loss: {running_loss/batch_count:.4f}")
    print(f"   Final Val Loss:   {final_val_loss:.4f}")
    print(f"   Epoch Time: {elapsed_time/60:.2f} minutes")
    print(f"{'='*80}\n")
    
    validation_history.append(final_val_loss)
    
    # Log final results
    with open(cfg.validation_log, 'a', encoding='utf-8') as f:
        f.write(f"\n{'='*80}\n")
        f.write(f"EPOCH {epoch_num} COMPLETE - {datetime.now()}\n")
        f.write(f"Final Train Loss: {running_loss/batch_count:.4f}\n")
        f.write(f"Final Val Loss: {final_val_loss:.4f}\n")
        f.write(f"Epoch Time: {elapsed_time/60:.2f} minutes ({elapsed_time/3600:.2f} hours)\n")
        f.write(f"Best Val Loss (this epoch): {min(validation_history):.4f}\n")
        f.write(f"{'='*80}\n\n")
    
    return running_loss / batch_count, final_val_loss, elapsed_time

# ============================================================================
# SAVE CHECKPOINT
# ============================================================================

def save_checkpoint(model, epoch_num, train_loss, val_loss, cfg):
    """Save model checkpoint"""
    
    cfg.output_dir.mkdir(exist_ok=True, parents=True)
    
    print(f"\n{'='*80}")
    print(f"💾 SAVING EPOCH {epoch_num} CHECKPOINT")
    print(f"{'='*80}")
    
    # Merge LoRA with base model
    if cfg.use_lora and PEFT_AVAILABLE:
        print(f"📦 Merging LoRA with base model...")
        merged_model = model.merge_and_unload()
        state_dict = merged_model.state_dict()
    else:
        state_dict = model.state_dict()
    
    # Save merged checkpoint
    checkpoint_path = cfg.output_dir / f"instruction_tuned_curated_epoch_{epoch_num}.pt"
    torch.save({
        'model_state_dict': state_dict,
        'epoch': epoch_num,
        'train_loss': train_loss,
        'val_loss': val_loss,
        'timestamp': datetime.now().isoformat(),
        'config': {
            'lora_r': cfg.lora_r,
            'lora_alpha': cfg.lora_alpha,
            'learning_rate': cfg.learning_rate,
            'dataset': str(cfg.data_file.name)
        }
    }, checkpoint_path)
    
    # Get file size
    checkpoint_size = checkpoint_path.stat().st_size / (1024**2)
    
    print(f"✅ Epoch {epoch_num} checkpoint saved: {checkpoint_path}")
    print(f"   Size: {checkpoint_size:.1f} MB")
    print(f"   Train Loss: {train_loss:.4f}")
    print(f"   Val Loss: {val_loss:.4f}")
    
    # Auto-download on Colab
    if IN_COLAB and cfg.auto_download_checkpoints:
        print(f"\n📥 Auto-downloading checkpoint to local machine...")
        try:
            files.download(str(checkpoint_path))
            print(f"✅ Download initiated: {checkpoint_path.name}")
        except Exception as e:
            print(f"⚠️  Download failed: {e}")
    
    print(f"{'='*80}\n")
    
    return checkpoint_path

# ============================================================================
# MAIN TRAINING FUNCTION
# ============================================================================

def continue_instruction_tuning():
    """Continue instruction tuning from epoch 1"""
    
    cfg = ContinuedTrainingConfig()
    
    print("\n" + "="*80)
    print("  🎓 CONTINUING INSTRUCTION TUNING — Epochs 2-3")
    print("="*80)
    print(f"\n📋 Configuration:")
    print(f"   Resuming from: {cfg.checkpoint_path.name}")
    print(f"   Dataset: {cfg.data_file.name}")
    print(f"   Epochs: {cfg.start_epoch} to {cfg.start_epoch + cfg.num_additional_epochs - 1}")
    print(f"   Validation Split: {cfg.validation_split*100:.0f}%")
    print(f"   Learning Rate: {cfg.learning_rate}")
    print(f"   Batch Size: {cfg.batch_size}")
    print(f"   Gradient Accumulation: {cfg.gradient_accumulation_steps}")
    print(f"   Effective Batch Size: {cfg.batch_size * cfg.gradient_accumulation_steps}")
    
    # Check checkpoint exists
    if not cfg.checkpoint_path.exists():
        print(f"\n❌ Checkpoint not found: {cfg.checkpoint_path}")
        return
    
    # Check data file
    if not cfg.data_file.exists():
        print(f"\n❌ Data file not found: {cfg.data_file}")
        return
    
    # Load model
    model, device = load_model_with_lora(cfg.checkpoint_path, cfg)
    
    # Load dataset (same split as before for consistency)
    tokenizer = get_tokenizer()
    full_dataset = InstructionDataset(cfg.data_file, tokenizer, cfg.max_length)
    
    train_size = int((1 - cfg.validation_split) * len(full_dataset))
    val_size = len(full_dataset) - train_size
    
    train_dataset, val_dataset = random_split(
        full_dataset, 
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42)  # Same seed for same split
    )
    
    print(f"\n📊 Dataset Split:")
    print(f"   Training: {len(train_dataset):,}")
    print(f"   Validation: {len(val_dataset):,}")
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=cfg.batch_size, 
        shuffle=True,
        num_workers=0
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=0
    )
    
    # Setup optimizer
    optimizer = AdamW(
        model.parameters(), 
        lr=cfg.learning_rate, 
        weight_decay=cfg.weight_decay
    )
    
    # Initialize log files
    with open(cfg.log_file, 'w', encoding='utf-8') as f:
        f.write(f"Continued Instruction Tuning (Epochs {cfg.start_epoch}-{cfg.start_epoch + cfg.num_additional_epochs - 1}) - {datetime.now()}\n")
        f.write(f"{'='*80}\n")
        f.write(f"Resuming from: {cfg.checkpoint_path.name}\n")
        f.write(f"Dataset: {cfg.data_file.name}\n")
        f.write(f"{'='*80}\n\n")
    
    with open(cfg.validation_log, 'w', encoding='utf-8') as f:
        f.write(f"Continued Training Validation Log - {datetime.now()}\n")
        f.write(f"{'='*80}\n\n")
    
    # Train for additional epochs
    total_start_time = time.time()
    all_checkpoints = []
    
    for epoch_offset in range(cfg.num_additional_epochs):
        epoch_num = cfg.start_epoch + epoch_offset
        
        # Train epoch
        train_loss, val_loss, epoch_time = train_epoch(
            model, train_loader, val_loader, optimizer, device, cfg, epoch_num
        )
        
        # Save checkpoint
        checkpoint_path = save_checkpoint(model, epoch_num, train_loss, val_loss, cfg)
        all_checkpoints.append((epoch_num, checkpoint_path, train_loss, val_loss))
        
        # Run final inference examples for this epoch
        if cfg.test_every_validation:
            tokenizer = get_tokenizer()
            print(f"\n{'='*80}")
            print(f"📊 FINAL INFERENCE FOR EPOCH {epoch_num}")
            print(f"{'='*80}")
            run_inference_examples(model, tokenizer, device, cfg, epoch_num, "FINAL")
    
    total_time = time.time() - total_start_time
    
    # Final summary
    print(f"\n{'='*80}")
    print(f"  ✅ CONTINUED TRAINING COMPLETE")
    print(f"{'='*80}")
    print(f"\n📊 All Epochs Summary:")
    for epoch, path, train_loss, val_loss in all_checkpoints:
        print(f"   Epoch {epoch}: Train={train_loss:.4f} | Val={val_loss:.4f}")
        print(f"      → {path.name}")
    
    print(f"\n⏱️  Total Training Time: {total_time/60:.2f} minutes ({total_time/3600:.2f} hours)")
    print(f"\n💾 Saved Checkpoints:")
    for _, path, _, _ in all_checkpoints:
        print(f"   ✓ {path}")
    
    print(f"\n📝 Logs:")
    print(f"   ✓ Training: {cfg.log_file}")
    print(f"   ✓ Validation: {cfg.validation_log}")
    
    print(f"\n{'='*80}\n")

# ============================================================================
# ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    continue_instruction_tuning()

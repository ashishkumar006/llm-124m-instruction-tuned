"""
Instruction Tuning Script with LoRA for 124M Model
Modified for curated dataset with validation tracking
- 1 epoch with 5% validation split
- Validation loss tracked at regular intervals
- Comprehensive logging with tqdm
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
sys.path.insert(0, str(Path(__file__).parent.parent / "llm_124m"))
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

# ============================================================================
# CONFIGURATION
# ============================================================================

class InstructionConfig:
    # Training hyperparameters
    num_epochs = 1  # Single epoch as requested
    learning_rate = 2e-5
    batch_size = 4
    gradient_accumulation_steps = 4  # Effective batch size = 16
    max_length = 2048
    weight_decay = 0.01
    warmup_steps = 50
    max_grad_norm = 1.0
    
    # Validation settings
    validation_split = 0.05  # 5% for validation
    validation_interval = 50  # Validate every 50 batches
    
    # LoRA configuration
    use_lora = True
    lora_r = 8
    lora_alpha = 16
    lora_dropout = 0.1
    
    # Paths
    checkpoint_path = Path(__file__).parent.parent / "checkpoint_step_7629.pt"
    data_file = Path(__file__).parent / "curated_dataset_english_only.json"  # Use English-only dataset
    output_dir = Path(__file__).parent.parent
    log_file = Path(__file__).parent / "training_log_curated.txt"
    validation_log = Path(__file__).parent / "validation_log_curated.txt"
    inference_log = Path(__file__).parent / "inference_progress_curated.txt"
    
    # Inference settings
    test_every_validation = True  # Test on validation set
    inference_temperature = 0.8
    inference_max_tokens = 150

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
    """Load model and add LoRA adapters"""
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n🔧 Device: {device}")
    
    # Load base model
    print(f"📦 Loading checkpoint from {checkpoint_path}...")
    config_wrapped = ConfigWrapper(config)
    model = LanguageModel(config_wrapped)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Remove torch.compile wrapper if present
    state_dict = checkpoint['model_state_dict']
    if any(k.startswith('_orig_mod.') for k in state_dict.keys()):
        print("   Removing torch.compile wrapper...")
        state_dict = {k.replace('_orig_mod.', ''): v for k, v in state_dict.items()}
    
    model.load_state_dict(state_dict)
    print("✅ Model loaded successfully")
    
    # Add compatibility methods for PEFT
    if not hasattr(model, 'prepare_inputs_for_generation'):
        model.prepare_inputs_for_generation = lambda *args, **kwargs: {}
    if not hasattr(model, 'get_input_embeddings'):
        model.get_input_embeddings = lambda: model.token_embed
    if not hasattr(model, 'set_input_embeddings'):
        model.set_input_embeddings = lambda x: setattr(model, 'token_embed', x)
    
    # Add LoRA
    if cfg.use_lora and PEFT_AVAILABLE:
        print("\n🔗 Adding LoRA adapters...")
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

def train_with_validation(model, train_loader, val_loader, optimizer, device, cfg):
    """Train with periodic validation"""
    
    model.train()
    training_history = []
    validation_history = []
    
    total_batches = len(train_loader)
    start_time = time.time()
    
    print(f"\n{'='*80}")
    print(f"  🚀 TRAINING WITH VALIDATION TRACKING")
    print(f"{'='*80}")
    print(f"  Total batches: {total_batches}")
    print(f"  Validation every: {cfg.validation_interval} batches")
    print(f"  Expected validation checks: {total_batches // cfg.validation_interval + 1}")
    print(f"{'='*80}\n")
    
    progress_bar = tqdm(train_loader, desc="Training", unit="batch")
    
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
        
        # Validation checkpoint
        if (batch_idx + 1) % cfg.validation_interval == 0:
            elapsed_time = time.time() - start_time
            
            print(f"\n{'─'*80}")
            print(f"📊 VALIDATION CHECKPOINT - Batch {batch_idx+1}/{total_batches}")
            print(f"   Time elapsed: {elapsed_time/60:.2f} minutes")
            
            # Evaluate on validation set
            val_loss = evaluate_validation(model, val_loader, device, cfg)
            
            # Record
            checkpoint_info = {
                'batch': batch_idx + 1,
                'train_loss': avg_train_loss,
                'val_loss': val_loss,
                'time_minutes': elapsed_time / 60
            }
            
            training_history.append(avg_train_loss)
            validation_history.append(val_loss)
            
            print(f"   Train Loss: {avg_train_loss:.4f}")
            print(f"   Val Loss:   {val_loss:.4f}")
            print(f"   {'✓ Improving' if len(validation_history) < 2 or val_loss < min(validation_history[:-1]) else '⚠ Not improving'}")
            print(f"{'─'*80}\n")
            
            # Log to file
            with open(cfg.validation_log, 'a', encoding='utf-8') as f:
                f.write(f"{datetime.now()} | Batch {batch_idx+1:4d}/{total_batches} | "
                       f"Train: {avg_train_loss:.4f} | Val: {val_loss:.4f} | "
                       f"Time: {elapsed_time/60:.2f}m\n")
    
    # Final validation
    print(f"\n{'='*80}")
    print("📊 FINAL VALIDATION")
    final_val_loss = evaluate_validation(model, val_loader, device, cfg)
    elapsed_time = time.time() - start_time
    
    print(f"   Final Train Loss: {running_loss/batch_count:.4f}")
    print(f"   Final Val Loss:   {final_val_loss:.4f}")
    print(f"   Total Time: {elapsed_time/60:.2f} minutes")
    print(f"{'='*80}\n")
    
    validation_history.append(final_val_loss)
    
    # Log final results
    with open(cfg.validation_log, 'a', encoding='utf-8') as f:
        f.write(f"\n{'='*80}\n")
        f.write(f"FINAL RESULTS - {datetime.now()}\n")
        f.write(f"Final Train Loss: {running_loss/batch_count:.4f}\n")
        f.write(f"Final Val Loss: {final_val_loss:.4f}\n")
        f.write(f"Total Time: {elapsed_time/60:.2f} minutes ({elapsed_time/3600:.2f} hours)\n")
        f.write(f"Validation checks: {len(validation_history)}\n")
        f.write(f"Best Val Loss: {min(validation_history):.4f}\n")
        f.write(f"{'='*80}\n\n")
    
    return running_loss / batch_count, validation_history

# ============================================================================
# SAVE CHECKPOINT
# ============================================================================

def save_checkpoint(model, loss, val_loss, cfg):
    """Save model checkpoint with both merged and LoRA-only versions"""
    
    cfg.output_dir.mkdir(exist_ok=True, parents=True)
    
    print(f"\n{'='*80}")
    print("💾 SAVING CHECKPOINTS")
    print(f"{'='*80}")
    
    # Save LoRA adapters separately
    if cfg.use_lora and PEFT_AVAILABLE:
        adapter_path = cfg.output_dir / "lora_adapters_curated"
        model.save_pretrained(adapter_path)
        print(f"✅ LoRA adapters saved to: {adapter_path}")
        print(f"   Size: ~491.5K parameters (LoRA only)")
        
        # Save full merged model
        print(f"\n📦 Merging LoRA with base model...")
        merged_model = model.merge_and_unload()
        state_dict = merged_model.state_dict()
    else:
        state_dict = model.state_dict()
    
    # Save merged checkpoint
    checkpoint_path = cfg.output_dir / "instruction_tuned_curated_epoch_1.pt"
    torch.save({
        'model_state_dict': state_dict,
        'epoch': 1,
        'train_loss': loss,
        'val_loss': val_loss,
        'timestamp': datetime.now().isoformat(),
        'config': {
            'lora_r': cfg.lora_r,
            'lora_alpha': cfg.lora_alpha,
            'learning_rate': cfg.learning_rate,
            'dataset': str(cfg.data_file.name)
        }
    }, checkpoint_path)
    
    # Get file sizes
    checkpoint_size = checkpoint_path.stat().st_size / (1024**2)
    
    print(f"✅ Full model checkpoint saved to: {checkpoint_path}")
    print(f"   Size: {checkpoint_size:.1f} MB (merged LoRA + base)")
    print(f"\n{'='*80}\n")
    
    return checkpoint_path, adapter_path if cfg.use_lora else None

# ============================================================================
# MAIN TRAINING FUNCTION
# ============================================================================

def instruction_tune():
    """Main instruction tuning pipeline"""
    
    cfg = InstructionConfig()
    
    print("\n" + "="*80)
    print("  🎓 INSTRUCTION TUNING — 124M Model (Curated Dataset)")
    print("="*80)
    print(f"\n📋 Configuration:")
    print(f"   Dataset: {cfg.data_file.name}")
    print(f"   Epochs: {cfg.num_epochs}")
    print(f"   Validation Split: {cfg.validation_split*100:.0f}%")
    print(f"   Validation Interval: {cfg.validation_interval} batches")
    print(f"   Learning Rate: {cfg.learning_rate}")
    print(f"   Batch Size: {cfg.batch_size}")
    print(f"   Gradient Accumulation: {cfg.gradient_accumulation_steps}")
    print(f"   Effective Batch Size: {cfg.batch_size * cfg.gradient_accumulation_steps}")
    print(f"   Max Sequence Length: {cfg.max_length}")
    print(f"   LoRA: r={cfg.lora_r}, alpha={cfg.lora_alpha}")
    
    # Check data file
    if not cfg.data_file.exists():
        print(f"\n❌ Data file not found: {cfg.data_file}")
        return
    
    # Load model
    model, device = load_model_with_lora(cfg.checkpoint_path, cfg)
    
    # Load dataset
    tokenizer = get_tokenizer()
    full_dataset = InstructionDataset(cfg.data_file, tokenizer, cfg.max_length)
    
    # Split into train and validation (95% / 5%)
    train_size = int((1 - cfg.validation_split) * len(full_dataset))
    val_size = len(full_dataset) - train_size
    
    train_dataset, val_dataset = random_split(
        full_dataset, 
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42)  # Reproducible split
    )
    
    print(f"\n📊 Dataset Split:")
    print(f"   Total examples: {len(full_dataset):,}")
    print(f"   Training: {len(train_dataset):,} ({100*train_size/len(full_dataset):.1f}%)")
    print(f"   Validation: {len(val_dataset):,} ({100*val_size/len(full_dataset):.1f}%)")
    
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
    
    print(f"\n📦 Batches:")
    print(f"   Training batches: {len(train_loader)}")
    print(f"   Validation batches: {len(val_loader)}")
    print(f"   Expected validation checks: {len(train_loader) // cfg.validation_interval + 1}")
    
    # Estimate time
    estimated_time_minutes = (len(train_loader) * 0.3)  # ~0.3 min per batch estimate
    print(f"\n⏱️  Estimated Training Time: {estimated_time_minutes:.1f} minutes ({estimated_time_minutes/60:.2f} hours)")
    
    # Setup optimizer
    optimizer = AdamW(
        model.parameters(), 
        lr=cfg.learning_rate, 
        weight_decay=cfg.weight_decay
    )
    
    # Initialize log files
    cfg.log_file.parent.mkdir(exist_ok=True, parents=True)
    with open(cfg.log_file, 'w', encoding='utf-8') as f:
        f.write(f"Instruction Tuning (Curated Dataset) - {datetime.now()}\n")
        f.write(f"{'='*80}\n")
        f.write(f"Dataset: {cfg.data_file.name}\n")
        f.write(f"Total examples: {len(full_dataset):,}\n")
        f.write(f"Training: {len(train_dataset):,} | Validation: {len(val_dataset):,}\n")
        f.write(f"LoRA: r={cfg.lora_r}, alpha={cfg.lora_alpha}\n")
        f.write(f"Learning rate: {cfg.learning_rate}\n")
        f.write(f"{'='*80}\n\n")
    
    with open(cfg.validation_log, 'w', encoding='utf-8') as f:
        f.write(f"Validation Tracking - {datetime.now()}\n")
        f.write(f"{'='*80}\n")
        f.write(f"Validation every {cfg.validation_interval} batches\n")
        f.write(f"{'='*80}\n\n")
    
    # Training
    start_time = time.time()
    train_loss, val_history = train_with_validation(
        model, train_loader, val_loader, optimizer, device, cfg
    )
    training_time = time.time() - start_time
    
    # Save checkpoint
    checkpoint_path, lora_path = save_checkpoint(model, train_loss, val_history[-1], cfg)
    
    # Final summary
    print(f"\n{'='*80}")
    print(f"  ✅ TRAINING COMPLETE")
    print(f"{'='*80}")
    print(f"\n📊 Results:")
    print(f"   Final Train Loss: {train_loss:.4f}")
    print(f"   Final Val Loss: {val_history[-1]:.4f}")
    print(f"   Best Val Loss: {min(val_history):.4f}")
    print(f"   Training Time: {training_time/60:.2f} minutes ({training_time/3600:.2f} hours)")
    print(f"   Validation Checks: {len(val_history)}")
    
    print(f"\n💾 Saved Files:")
    print(f"   ✓ Full model: {checkpoint_path}")
    print(f"     Size: ~435 MB (merged LoRA + base)")
    if lora_path:
        print(f"   ✓ LoRA adapters: {lora_path}")
        print(f"     Size: ~151 MB (LoRA matrices only)")
    print(f"   ✓ Training log: {cfg.log_file}")
    print(f"   ✓ Validation log: {cfg.validation_log}")
    
    print(f"\n{'='*80}\n")

# ============================================================================
# ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    instruction_tune()

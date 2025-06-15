import logging
from typing import Dict, Any

import torch
from torch.utils.data import Dataset
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import lightning as L
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.data_process.process_data import prepare_dataset_latent_sft
from src.latent_reasoner.model import LatentReasoner
from src.train.utils import is_rank_zero, setup_special_tokens

class ModelLightningModule(L.LightningModule):
    """PyTorch Lightning module for training a Language Model with SFT."""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        self.config = config
        self.tokenizer = AutoTokenizer.from_pretrained(config["model"]["base_model_name_or_path"])
        self.save_hyperparameters(config)
        
        # Initialize model
        if self.config["model"]["is_latent_reasoner"]:
            # For latent reasoner models, use the LatentReasoner class
            self.model = LatentReasoner.from_pretrained(config["model"]["base_model_name_or_path"])
        else:
            self.model = AutoModelForCausalLM.from_pretrained(self.config["model"]["base_model_name_or_path"])
        
        # Setup special tokens
        self.model, self.tokenizer = setup_special_tokens(
            model=self.model, 
            tokenizer=self.tokenizer,
            is_latent_reasoner=self.config["model"]["is_latent_reasoner"])

        # Store training config for optimizer setup
        self.learning_rate = float(self.config["training"]["learning_rate"])
        self.weight_decay = self.config["training"]["weight_decay"]
        self.lr_scheduler_step_size = self.config["training"]["lr_scheduler_step_size"]
        self.lr_scheduler_gamma = self.config["training"]["lr_scheduler_gamma"]
    
    def forward(self, input_ids, attention_mask, labels=None):
        """Forward pass through the model."""
        if self.config["model"]["is_latent_reasoner"]:
            return self.model.sft_forward(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
        else:            
            return self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )

    def training_step(self, batch, batch_idx):
        """Training step."""
        outputs = self.forward(
            input_ids=batch['input_ids'],
            attention_mask=batch['attention_mask'],
            labels=batch['labels']
        )
        
        loss = outputs.loss
        
        # Log metrics
        self.log('train_loss', loss, on_step=True, on_epoch=True)
        
        return loss

    def validation_step(self, batch, batch_idx):
        """Validation step."""
        outputs = self.forward(
            input_ids=batch['input_ids'],
            attention_mask=batch['attention_mask'],
            labels=batch['labels']
        )
        
        loss = outputs.loss
        
        # Log metrics
        self.log('val_loss', loss, on_epoch=True, on_step=False)
        
        return loss

    def configure_optimizers(self):
        """Configure optimizer and learning rate scheduler."""
        # Create optimizer
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay
        )
        
        # Create scheduler
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=self.config["training"]["lr_scheduler_step_size"],
            gamma=self.config["training"]["lr_scheduler_gamma"]
        )
        
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1,
            },
        }


class SFTDataset(Dataset):
    """Dataset wrapper for Lightning DataLoader."""
    
    def __init__(self, dataset):
        self.dataset = dataset
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        item = self.dataset[idx]
        return {
            'input_ids': torch.tensor(item['input_ids'], dtype=torch.long),
            'attention_mask': torch.tensor(item['attention_mask'], dtype=torch.long),
            'labels': torch.tensor(item['labels'], dtype=torch.long)
        }
    
def collate_fn(batch):
    """Collate function to pad sequences to the same length."""
    # Get max length in the batch
    max_len = max(len(item['input_ids']) for item in batch)
    
    # Pad sequences
    padded_batch = {
        'input_ids': [],
        'attention_mask': [],
        'labels': []
    }
    
    for item in batch:
        input_len = len(item['input_ids'])
        pad_len = max_len - input_len
        
        # Pad input_ids and attention_mask with pad_token_id (0)
        padded_batch['input_ids'].append(
            F.pad(item['input_ids'], (0, pad_len), value=0)
        )
        padded_batch['attention_mask'].append(
            F.pad(item['attention_mask'], (0, pad_len), value=0)
        )
        # Pad labels with -100 (ignore_index)
        padded_batch['labels'].append(
            F.pad(item['labels'], (0, pad_len), value=-100)
        )
    
    # Stack tensors
    return {
        'input_ids': torch.stack(padded_batch['input_ids']),
        'attention_mask': torch.stack(padded_batch['attention_mask']),
        'labels': torch.stack(padded_batch['labels'])
    }


class SFTDataModule(L.LightningDataModule):
    """Lightning DataModule for SFT training."""
    
    def __init__(self, config: Dict[str, Any], tokenizer):
        super().__init__()
        self.config = config
        self.tokenizer = tokenizer
        self.batch_size = config["training"]["per_device_train_batch_size"]
        self.eval_batch_size = config["training"]["per_device_eval_batch_size"]
        self.is_latent_reasoner = config["model"]["is_latent_reasoner"]
        self.current_epoch_num = 0  # Track current epoch for dataset preprocessing
        # If latent reasoning is enabled, prepare the dataset with latent tokens
        if self.is_latent_reasoner:
            self.num_tokens_per_latent = self.config["training"]["num_tokens_per_latent"]
            self.add_num_latents_per_update = self.config["training"]["add_num_latents_per_update"]
            assert self.num_tokens_per_latent > 0, "num_tokens_per_latent must be greater than 0."
            assert self.add_num_latents_per_update > 0, "add_num_latents_per_update must be greater than 0."
        else:
            # For standard SFT, no latent steps are used
            self.num_tokens_per_latent = None        

    def setup(self, stage: str, update_cycle: int = 0):
        """Setup datasets."""
        # Update current epoch number
        self.update_cycle = update_cycle
        # Prepare the dataset
        data = prepare_dataset_latent_sft(dataset_name=self.config["dataset"]["name"], 
                                          num_examples=self.config["dataset"]["num_examples"],
                                          tokenizer=self.tokenizer, 
                                          seed=self.config["training"]["seed"],
                                          num_tokens_per_latent=self.num_tokens_per_latent,
                                          add_num_latents_per_update=self.add_num_latents_per_update,
                                          update_cycle=update_cycle)
        
        self.train_dataset = SFTDataset(data["train"])
        self.val_dataset = SFTDataset(data["validation"])

        # 3. Show one training example for sanity-check
        if len(self.train_dataset) and is_rank_zero():
            sample = self.train_dataset[0]

            # decode the input text so it’s readable
            decoded_prompt = self.tokenizer.decode(
                sample["input_ids"],
                skip_special_tokens=False,
                clean_up_tokenization_spaces=True,
            )

            print(f"\n=== SFT Data Check (Update Cycle: {update_cycle}) ===")
            print("Prompt:\n", decoded_prompt)
            # if you store the answer/label under another key, adjust here
            if "labels" in sample:
                print("\nLabel IDs:", sample["labels"])
            print("====================================\n")        

    def update_dataset(self, update_cycle: int):
        """Re-setup datasets for a new update_cycle with different preprocessing."""
        self.setup("fit", update_cycle)

    def train_dataloader(self):
        """Return training dataloader."""
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            collate_fn=collate_fn,
            num_workers=4,
            pin_memory=True
        )

    def val_dataloader(self):
        """Return validation dataloader."""
        return DataLoader(
            self.val_dataset,
            batch_size=self.eval_batch_size,
            shuffle=False,
            collate_fn=collate_fn,
            num_workers=4,
            pin_memory=True
        )


class GenerateSamplesCallback(L.Callback):
    """
    After each validation epoch, generate answers for the *first* `num_samples`
    examples in the validation set and log them.
    """
    def __init__(self, tokenizer, num_samples: int = 2, is_latent_reasoner: bool = False):
        super().__init__()
        self.tokenizer = tokenizer
        self.num_samples = num_samples
        self.is_latent_reasoner = is_latent_reasoner

    @torch.inference_mode()
    def on_validation_epoch_end(
        self,
        trainer: L.Trainer,
        pl_module: L.LightningModule
    ):
        # Log for the main process only
        if trainer.global_rank != 0:
            return
        
        current_step = trainer.global_step
        add_num_latents_per_update = trainer.datamodule.config["training"]["add_num_latents_per_update"]
        dataset_refresh_every_n_steps = trainer.datamodule.config["training"]["dataset_refresh_every_n_steps"]
        update_cycle = current_step // dataset_refresh_every_n_steps
        total_latents = update_cycle * add_num_latents_per_update
        
        pl_module.eval()

        # Grab val_dataset directly from the datamodule
        val_dataset = trainer.datamodule.val_dataset
        if val_dataset is None:
            logging.warning("val_dataset is not available yet.")
            return

        # Always take the first `num_samples` examples (or fewer if the dataset is smaller)
        indices = list(range(min(self.num_samples, len(val_dataset))))

        for n, idx in enumerate(indices, start=1):
            sample = val_dataset[idx]

            # sample["input_ids"] is a tensor → convert to list[int]
            full_ids_tensor = sample["input_ids"]
            full_ids = full_ids_tensor.tolist()

            # Cut just before the answer starts
            try:
                cut_idx = full_ids.index(self.tokenizer.start_latent_token_id) # first <|start-latent|>
            except ValueError:
                cut_idx = full_ids.index(self.tokenizer.start_think_token_id) # first <think>

            prompt_ids = full_ids[:cut_idx]

            # back to tensor for generation
            input_ids = torch.tensor(prompt_ids, dtype=torch.long, device=pl_module.device).unsqueeze(0)
            attention_mask = torch.ones_like(input_ids)

            generated_ids = pl_module.model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                num_latent_steps=total_latents,
                max_new_tokens=512,
                do_sample=False
            )

            question = self.tokenizer.decode(
                input_ids[0].tolist(),
                skip_special_tokens=False,
                clean_up_tokenization_spaces=True
            )
            seq = generated_ids[0] # first (and only) sequence
            if isinstance(seq, torch.Tensor):
                seq = seq.tolist()
            # handle the "double bracket" case: [[1,2,3,…]]
            if isinstance(seq, list) and seq and isinstance(seq[0], list):
                seq = seq[0]

            answer = self.tokenizer.decode(
                seq,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=True,
            )

            step = trainer.global_step
            logging.info(f"[step {step}] [VAL Question {n}] ❓ {question}")
            logging.info(f"[step {step}] [VAL Answer   {n}] 🤖 {answer}")


class DatasetRefreshCallback(L.Callback):
    """
    Used only for SFT latent reasoning models.
    Callback to refresh the dataset preprocessing after every X training steps.
    This allows for dynamic changes to the dataset based on the current step.
    """
    def __init__(self, dataset_refresh_every_n_steps: int):
        super().__init__()
        self.dataset_refresh_every_n_steps = dataset_refresh_every_n_steps
        self.last_refresh_step = 0

    def _reset_optimizer_state(self, trainer: L.Trainer):
        """
        Zero-out the internal state of every optimizer attached to the trainer.
        Keeps learning-rate, parameter groups, and scheduler bindings intact.
        """
        for opt in trainer.optimizers:
            opt.state.clear()  # ← momentum, exp. moving avgs, etc.
        logging.info("Optimizer state has been reset after dataset refresh.")

    def _reset_lr_scheduler_state(self, trainer: L.Trainer, pl_module: L.LightningModule):
        """
        Reset the learning rate scheduler state and restore initial learning rate.
        This ensures consistent learning dynamics after dataset refresh.
        """
        # Get the initial learning rate from the module config
        initial_lr = pl_module.learning_rate
        
        # Reset learning rate in all parameter groups
        for opt in trainer.optimizers:
            for param_group in opt.param_groups:
                param_group['lr'] = initial_lr
        
        # Reset scheduler state if schedulers exist
        if hasattr(trainer, 'lr_scheduler_configs') and trainer.lr_scheduler_configs:
            for lr_scheduler_config in trainer.lr_scheduler_configs:
                scheduler = lr_scheduler_config.scheduler
                
                # Reset scheduler state
                if hasattr(scheduler, 'last_epoch'):
                    scheduler.last_epoch = -1
                
                # For StepLR scheduler, reset the step count
                if hasattr(scheduler, '_step_count'):
                    scheduler._step_count = 0
                
                # Reset any internal state
                if hasattr(scheduler, 'state_dict'):
                    # Create a fresh scheduler with same parameters to get clean state
                    if isinstance(scheduler, torch.optim.lr_scheduler.StepLR):
                        fresh_scheduler = torch.optim.lr_scheduler.StepLR(
                            trainer.optimizers[0],  # assuming single optimizer
                            step_size=pl_module.lr_scheduler_step_size,
                            gamma=pl_module.lr_scheduler_gamma
                        )
                        scheduler.load_state_dict(fresh_scheduler.state_dict())
        
        logging.info(f"Learning rate scheduler state reset. Initial LR restored: {initial_lr}")

    def on_train_batch_end(self, trainer: L.Trainer, pl_module: L.LightningModule, *args, **kwargs):
        """Called at the end of each training batch to check if refresh is needed."""
        # Only refresh on the main process to avoid duplicate work
        if trainer.global_rank != 0:
            return
        
        current_step = trainer.global_step
        
        # Check if it's time to refresh the dataset
        if (current_step > 0 and 
            current_step % self.dataset_refresh_every_n_steps == 0 and 
            current_step > self.last_refresh_step):
            # Calculate current latent count based on steps
            add_num_latents_per_update = trainer.datamodule.config["training"]["add_num_latents_per_update"]
            # Calculate latents based on refresh cycles
            update_cycle = current_step // self.dataset_refresh_every_n_steps
            total_latents = update_cycle * add_num_latents_per_update
            
            # Log to wandb through the Lightning module
            pl_module.log('step/total_latents', total_latents, on_epoch=False, on_step=True)

            # Refresh the dataset preprocessing
            trainer.datamodule.update_dataset(update_cycle=update_cycle)
            
            # Reset the optimizer
            self._reset_optimizer_state(trainer)
            # Reset the learning rate scheduler state and restore initial LR
            self._reset_lr_scheduler_state(trainer, pl_module)

            # Update last refresh step
            self.last_refresh_step = current_step
            
            logging.info(f"Dataset preprocessing refreshed at step {current_step} (update cycle: {update_cycle})")

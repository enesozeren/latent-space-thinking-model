import logging
from typing import Dict, Any, Optional
import os
import shutil

import torch
from torch.utils.data import Dataset
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.data_process.process_data import prepare_dataset_sft
from src.latent_reasoner.model import LatentReasoner
from src.train.utils import is_rank_zero, setup_latent_tokens
from src.train.utils import count_max_total_latents

class ModelLightningModule(L.LightningModule):
    """PyTorch Lightning module for training a Language Model with SFT."""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        self.config = config
        self.save_hyperparameters(config)
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(config["model"]["base_model_name_or_path"])
        # Initialize model
        if self.config["model"]["is_latent_reasoner"]:
            # For latent reasoner models, use the LatentReasoner class
            self.model = LatentReasoner.from_pretrained(self.config["model"]["base_model_name_or_path"])
        else:
            self.model = AutoModelForCausalLM.from_pretrained(self.config["model"]["base_model_name_or_path"])
        
        # Setup special tokens
        self.model, self.tokenizer = setup_latent_tokens(
            model=self.model, 
            tokenizer=self.tokenizer,
            is_latent_reasoner=self.config["model"]["is_latent_reasoner"])

        # Enable gradient checkpointing for efficiency
        self.model.gradient_checkpointing_enable()

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
            self.start_num_latents = self.config["training"]["start_num_latents"]
            self.max_num_latents = self.config["training"]["max_num_latents"]
            self.num_tokens_per_latent = self.config["training"]["num_tokens_per_latent"]
            self.add_num_latents_per_update = self.config["training"]["add_num_latents_per_update"]
            assert self.num_tokens_per_latent > 0, "num_tokens_per_latent must be greater than 0."
            assert self.add_num_latents_per_update > 0, "add_num_latents_per_update must be greater than 0."
        else:
            # For standard SFT, no latent steps are used
            self.max_num_latents = None
            self.num_tokens_per_latent = None
            self.add_num_latents_per_update = None

    def setup(self, stage: str, next_epoch: int = 1):
        """Setup datasets."""
        # Update current epoch number
        update_cycle = next_epoch
        # Prepare the dataset
        data = prepare_dataset_sft(
            dataset_name=self.config["dataset"]["name"], 
            num_examples=self.config["dataset"]["num_examples"],
            tokenizer=self.tokenizer, 
            seed=self.config["training"]["seed"],
            is_latent_reasoner=self.is_latent_reasoner,
            start_num_latents=self.start_num_latents,
            num_tokens_per_latent=self.num_tokens_per_latent,
            add_num_latents_per_update=self.add_num_latents_per_update,
            update_cycle=update_cycle,
            max_num_latents=self.max_num_latents
        )

        self.train_dataset = SFTDataset(data["train"])
        self.val_dataset = SFTDataset(data["validation"])

        # 3. Show one training example for sanity-check
        if len(self.train_dataset) and is_rank_zero():
            sample = self.train_dataset[0]

            # decode the input text so it's readable
            decoded_prompt = self.tokenizer.decode(
                sample["input_ids"],
                skip_special_tokens=False,
                clean_up_tokenization_spaces=True,
            )

            print(f"\n=== SFT Data Check (Update Cycle: {update_cycle}) ===")
            print("Prompt:\n", decoded_prompt)
            torch.set_printoptions(threshold=float('inf'))
            print("\nLabel IDs:", sample["labels"])
            print("====================================\n")        

    def update_dataset(self, next_epoch: int):
        """Re-setup datasets for a new epoch with different preprocessing."""
        self.setup("fit", next_epoch)

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
    def __init__(self, num_samples: int, 
                 is_latent_reasoner: bool,
                 start_num_latents: Optional[int] = None,
                 add_num_latents_per_update: Optional[int] = None,
                 max_num_latents: Optional[int] = None):
        super().__init__()
        self.num_samples = num_samples
        self.is_latent_reasoner = is_latent_reasoner
        self.start_num_latents = start_num_latents
        self.add_num_latents_per_update = add_num_latents_per_update
        self.max_num_latents = max_num_latents

    @torch.inference_mode()
    def on_validation_epoch_end(
        self,
        trainer: L.Trainer,
        pl_module: L.LightningModule
    ):
        # Log for the main process only
        if trainer.global_rank != 0:
            return
        
        update_cycle = trainer.current_epoch + 1
        if self.is_latent_reasoner:
            total_latents = count_max_total_latents(
                start_num_latents=self.start_num_latents,
                add_num_latents_per_update=self.add_num_latents_per_update,
                update_cycle=update_cycle,
                max_num_latents=self.max_num_latents
            )
        
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

            # Convert token IDs to string
            full_text = pl_module.tokenizer.decode(full_ids)

            # Determine which token string to search for
            if self.is_latent_reasoner and total_latents > 0:
                target_token = "<|start-latent|>"
            else:
                target_token = "<think>"

            # Find all occurrences of the target token in the text
            token_positions = []
            start_pos = 0
            while True:
                pos = full_text.find(target_token, start_pos)
                if pos == -1:
                    break
                token_positions.append(pos)
                start_pos = pos + 1
            
            prompt_finish_idx = token_positions[2] # since there are 2 <think> or <|start-latent|> tokens in the system prompt

            # Get the text up to the third token
            prompt_text = full_text[:prompt_finish_idx]

            # Convert back to token IDs
            prompt_ids = pl_module.tokenizer.encode(prompt_text, add_special_tokens=False)

            # back to tensor for generation
            input_ids = torch.tensor(prompt_ids, dtype=torch.long, device=pl_module.device).unsqueeze(0)
            attention_mask = torch.ones_like(input_ids)

            if self.is_latent_reasoner:
                generated_ids = pl_module.model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    num_latent_steps=total_latents,
                    max_new_tokens=2048,
                    do_sample=False
                )
            else:
                generated_ids = pl_module.model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=2048,
                    do_sample=False
                )
                # For non-latent reasoner, we need to remove the prompt from generated_ids
                # Since we're now working with string-based cutting, we use the original cut approach
                # but need to find the equivalent cut_idx in the original token sequence
                cut_idx = len(prompt_ids)
                generated_ids = generated_ids[:, cut_idx:]

            question = pl_module.tokenizer.decode(
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

            answer = pl_module.tokenizer.decode(
                seq,
                skip_special_tokens=False,
                clean_up_tokenization_spaces=True,
            )

            step = trainer.global_step
            logging.info(f"[step {step}] [VAL Question {n}] {question}")
            logging.info(f"[step {step}] [VAL Answer   {n}] {answer}")


class DatasetRefreshCallback(L.Callback):
    """
    For latent-reasoner SFT training:
    – refresh the datamodule's preprocessing every epoch
    – hard-reset *all* optimisers and LR schedulers to their pristine state
    """
    def __init__(self, start_num_latents: Optional[int], add_num_latents_per_update: int, num_tokens_per_latent: int, max_num_latents: int):
        super().__init__()
        self.start_num_latents = start_num_latents
        self.add_num_latents_per_update = add_num_latents_per_update
        self.num_tokens_per_latent = num_tokens_per_latent
        self.max_num_latents = max_num_latents

    #  Helpers
    @staticmethod
    def _reset_optimizer_state(optimizer: torch.optim.Optimizer) -> None:
        """
        Wipes *all* per-parameter buffers (momentum, exp-avg, 8-bit stats, …)
        but keeps parameter groups & hyper-parameters intact.
        This is safe for Adam/AdamW, Lion, LAMB, SGD-momentum, fused & 8-bit
        variants – they lazily recreate state at the next step.
        """
        optimizer.state.clear()

        # Re-initialise if the optimiser exposes a helper (e.g. bitsandbytes)
        if hasattr(optimizer, "_optimizer__initialize"):
            optimizer._optimizer__initialize()

    @staticmethod
    def _reset_scheduler_state(scheduler: torch.optim.lr_scheduler._LRScheduler) -> None:
        """
        Generic reset that works for StepLR, CosineAnnealingLR, OneCycleLR,
        ReduceLROnPlateau, etc.
        """
        # Restore each param-group's LR to the original base LR
        for group, base_lr in zip(scheduler.optimizer.param_groups, scheduler.base_lrs):
            group["lr"] = base_lr

        # Lightning always calls scheduler.step() after optimiser.step().
        # Setting last_epoch = -1 and _step_count = 0 guarantees that the next
        # call behaves as if it's the very first scheduler step.
        scheduler.last_epoch = -1
        if hasattr(scheduler, "_step_count"):
            scheduler._step_count = 0

        # Special-case attributes for plateau schedulers
        if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            scheduler.best = float("inf")
            scheduler.num_bad_epochs = 0
            scheduler.cooldown_counter = 0

    #  Main hook
    def on_train_epoch_end(
        self,
        trainer: L.Trainer,
        pl_module: L.LightningModule,
        *args, **kwargs,
    ):
        # 1. Work out how many latents we had for the current epoch
        current_epoch = trainer.current_epoch + 1 # since epoch starts with 0
        logging.info(f"Current epoch which ends: {current_epoch}") 
        next_epoch = trainer.current_epoch + 2
        total_latents_this_epoch = count_max_total_latents(
            self.start_num_latents,
            add_num_latents_per_update=self.add_num_latents_per_update,
            update_cycle=current_epoch,
            max_num_latents=self.max_num_latents
        )
        pl_module.log(
            "step/total_latents",
            total_latents_this_epoch,
            on_epoch=True,
            on_step=False,
        )

        # 2. Refresh the dataset (every rank – Lightning will DDP-spawn)
        trainer.datamodule.update_dataset(next_epoch=next_epoch)

        # 3. Reset every optimiser
        for opt in trainer.optimizers:
            self._reset_optimizer_state(opt)

        # 4. Reset every LR scheduler
        sched_cfgs = getattr(trainer, "lr_schedulers",
                             getattr(trainer, "lr_scheduler_configs", []))
        for cfg in sched_cfgs:
            # cfg is a dict in new PL, an AttrDict-like object in old PL
            scheduler = cfg["scheduler"] if isinstance(cfg, dict) else cfg.scheduler
            self._reset_scheduler_state(scheduler)

        logging.info(f"Dataset + optimiser/scheduler reset (cycle {current_epoch}).")

        # 5. Make sure every rank arrives here before training resumes
        trainer.strategy.barrier()


class HFModelCheckpoint(ModelCheckpoint):
    """
    Latent-reasoner: keep every epoch (epoch00/, epoch01/, …).
    Regular model  : keep *one* best folder, named step{step:04d}/.
    """

    def __init__(self, output_dir: str, is_latent_reasoner: bool = False, **kwargs):
        ckpt_root = os.path.join(output_dir, "hf_checkpoints")

        if is_latent_reasoner:
            super().__init__(
                dirpath=ckpt_root,
                filename="{epoch:02d}",     # epoch00/, epoch01/, …
                every_n_epochs=1,
                save_top_k=-1,                   # keep all epochs
                **kwargs,
            )
        else:
            super().__init__(
                dirpath=ckpt_root,
                filename="{step:04d}",       # step0001/, …
                monitor="val_loss",
                mode="min",
                save_top_k=1,                    # keep only the best
                **kwargs,
            )

        self.is_latent_reasoner = is_latent_reasoner
        self._prev_best_dir: str | None = None   # track the folder to delete next time

    #  Replace the .ckpt write with HF’s save_pretrained + tokenizer.
    def _save_checkpoint(self, trainer, filepath: str) -> None:

        # Save only from rank 0
        if trainer.global_rank != 0:
            return

        # PL passes step0001.  Strip the extension → folder path.
        save_dir = os.path.splitext(filepath)[0]
        os.makedirs(save_dir, exist_ok=True)

        pl_module = trainer.lightning_module
        pl_module.model.save_pretrained(save_dir)
        pl_module.tokenizer.save_pretrained(save_dir)

        # regular-model cleanup: delete the prior "best" dir
        if not self.is_latent_reasoner and self._prev_best_dir and \
           os.path.isdir(self._prev_best_dir) and self._prev_best_dir != save_dir:
            try:
                shutil.rmtree(self._prev_best_dir)
            except OSError as exc:
                logging.warning(f"Couldn’t delete stale best dir {self._prev_best_dir}: {exc}")

        # remember current best
        if not self.is_latent_reasoner:
            self._prev_best_dir = save_dir

        # let PL know where the latest checkpoint lives (so resume works)
        self.last_model_path = save_dir

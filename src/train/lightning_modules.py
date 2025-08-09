import logging
from typing import Dict, Any, Optional
import os
import shutil

import numpy as np
import math
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
from src.eval.eval import extract_answer_from_response

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

        self.max_num_latents = config.get("training", {}).get("max_num_latents", 0)
        self.num_tokens_per_latent = config.get("training", {}).get("num_tokens_per_latent", 0)
        if self.is_latent_reasoner:
            assert self.num_tokens_per_latent > 0, "num_tokens_per_latent must be greater than 0."

    def setup(self, stage: str, total_num_latents: int = 1):
        """Setup datasets."""
        # Prepare the dataset
        data = prepare_dataset_sft(
            dataset_name=self.config["dataset"]["name"], 
            num_examples=self.config["dataset"]["num_examples"],
            tokenizer=self.tokenizer, 
            seed=self.config["training"]["seed"],
            is_latent_reasoner=self.is_latent_reasoner,
            num_tokens_per_latent=self.num_tokens_per_latent,
            max_num_latents=self.max_num_latents,
            total_num_latents=total_num_latents
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

            print(f"\n=== SFT Data Check (Total Numb of Latents: {total_num_latents}) ===")
            print("Prompt:\n", decoded_prompt)
            torch.set_printoptions(threshold=float('inf'))
            print("\nLabel IDs:", sample["labels"])
            print("====================================\n")        

    def update_dataset(self, total_num_latents: int):
        """Re-setup datasets for a new epoch with different preprocessing."""
        self.setup("fit", total_num_latents)

    def train_dataloader(self):
        """Return training dataloader."""
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            collate_fn=collate_fn,
            num_workers=32,
            pin_memory=True,
            persistent_workers=True
        )

    def val_dataloader(self):
        """Return validation dataloader."""
        return DataLoader(
            self.val_dataset,
            batch_size=self.eval_batch_size,
            shuffle=False,
            collate_fn=collate_fn,
            num_workers=32,
            pin_memory=True,
            persistent_workers=True
        )


class GenerateSamplesCallback(L.Callback):
    """
    After each validation epoch, generate answers for the *first* `num_samples`
    examples in the validation set and log them.
    """
    def __init__(self, num_samples: int, 
                 is_latent_reasoner: bool,
                 start_num_latents: Optional[int] = None,
                 add_latents_delta: Optional[int] = None,
                 max_num_latents: Optional[int] = None):
        super().__init__()
        self.num_samples = num_samples
        self.is_latent_reasoner = is_latent_reasoner
        self.start_num_latents = start_num_latents
        self.add_latents_delta = add_latents_delta
        self.max_num_latents = max_num_latents

    @torch.inference_mode()
    def on_validation_epoch_end(
        self,
        trainer: L.Trainer,
        pl_module: L.LightningModule
    ):
        num_batches_this_epoch = trainer.num_training_batches 

        # Log for the main process only
        if trainer.global_rank != 0 or math.isinf(num_batches_this_epoch):
            return

        current_batch = trainer.fit_loop.batch_idx
        current_epoch = trainer.current_epoch
        
        logging.info("Generate Samples Callback Triggered")
        logging.info(f"Current Epoch: {current_epoch} & Current Batch: {current_batch}")
        logging.info(f"Num Batches This Epoch: {num_batches_this_epoch}")
        if self.is_latent_reasoner:
            total_latents = count_max_total_latents(
                start_num_latents=self.start_num_latents,
                add_latents_delta=self.add_latents_delta,
                current_epoch=current_epoch,
                current_batch=current_batch,
                num_batches_this_epoch=num_batches_this_epoch,
                max_num_latents=self.max_num_latents
            )
            logging.info(f"Total Latents: {total_latents}")
        
        pl_module.eval()

        # Grab val_dataset directly from the datamodule
        val_dataset = trainer.datamodule.val_dataset
        if val_dataset is None:
            logging.warning("val_dataset is not available yet.")
            return

        # Always take the first `num_samples` examples (or fewer if the dataset is smaller)
        indices = list(range(min(self.num_samples, len(val_dataset))))

        # Track accuracy and length for samples
        correct_predictions = 0
        total_accuracy_samples = 0
        answer_token_length_list = []

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
                    max_new_tokens=1024,
                    do_sample=False
                )
            else:
                generated_ids = pl_module.model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=1024,
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

            answer_token_length_list.append(len(seq))

            # Calculate accuracy for all samples
            # Extract ground truth answer content from the original full text answer part
            gt_answer_content = extract_answer_from_response(full_text[prompt_finish_idx:])
            
            # Extract predicted answer content from the generated answer
            pred_answer_content = extract_answer_from_response(answer)
            
            # Check if both answers were found and match exactly
            # Empty string means no answer was found
            is_correct = (gt_answer_content != "" and 
                         pred_answer_content != "" and 
                         gt_answer_content == pred_answer_content)
            
            if is_correct:
                correct_predictions += 1
                
            total_accuracy_samples += 1
            
            logging.info(f"[Epoch {current_epoch} & Batch {current_batch}] [VAL Accuracy {n}] \Ground Truth: '{gt_answer_content}', Predicted: '{pred_answer_content}', Correct: {is_correct}")
            logging.info(f"[Epoch {current_epoch} & Batch {current_batch}] [VAL Question {n}] {question}")
            logging.info(f"[Epoch {current_epoch} & Batch {current_batch}] [VAL Answer   {n}] {answer}")
            logging.info(f"[Epoch {current_epoch} & Batch {current_batch}] [VAL Answer Length {n}] {len(seq)}")

        # Calculate and log final accuracy
        accuracy = correct_predictions / total_accuracy_samples
        logging.info(f"[Epoch {current_epoch} & Batch {current_batch}] [VAL ACCURACY] {correct_predictions}/{total_accuracy_samples} = {accuracy:.4f}")
        # Avg token lenght
        avg_num_tokens = np.mean(answer_token_length_list)
        logging.info(f"[Epoch {current_epoch} & Batch {current_batch}] [VAL Avg Answer Length] {avg_num_tokens}")

        # Log to wandb
        pl_module.log("step/val_accuracy", accuracy, on_epoch=True)
        pl_module.log("step/val_answer_token_len", avg_num_tokens, on_epoch=True)
        

class DatasetRefreshCallback(L.Callback):
    """
    For latent-reasoner SFT training:
    – refresh the datamodule's preprocessing every epoch
    – hard-reset *all* optimisers and LR schedulers to their pristine state
    """
    def __init__(self, start_num_latents: Optional[int], add_latents_delta: int, num_tokens_per_latent: int, max_num_latents: int):
        super().__init__()
        self.start_num_latents = start_num_latents
        self.add_latents_delta = add_latents_delta
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

    #  Main hook
    def on_train_batch_end(
        self,
        trainer: L.Trainer,
        pl_module: L.LightningModule,
        outputs, batch, batch_idx,
        *args, **kwargs,
    ):

        current_batch = batch_idx
        current_epoch = trainer.current_epoch
        num_batches_this_epoch = trainer.num_training_batches
        interval_to_add_latent = num_batches_this_epoch // self.add_latents_delta
    
        # Update the dataset if it is time
        if (current_batch) % interval_to_add_latent == 0:
            # Get the number of latent steps in this epoch
            logging.info("Dataset Refresh Callback Triggered")
            logging.info(f"Current epoch: {current_epoch}")
            logging.info(f"Current batch: {current_batch}")
            logging.info(f"Total numb of batches in this epoch: {num_batches_this_epoch}")
            total_num_latents = count_max_total_latents(
                self.start_num_latents,
                add_latents_delta=self.add_latents_delta,
                current_epoch=current_epoch,
                current_batch=current_batch,
                num_batches_this_epoch=num_batches_this_epoch,
                max_num_latents=self.max_num_latents
            )
            logging.info(f"Numb of latent steps in this cycle: {total_num_latents}")

            pl_module.log(
                "step/total_latents",
                total_num_latents,
                on_epoch=False,
                on_step=True,
            )

            # 2. Refresh the dataset (every rank – Lightning will DDP-spawn)
            trainer.datamodule.update_dataset(total_num_latents=total_num_latents)

            # 3. Reset optimiser
            trainer.strategy.setup_optimizers(trainer)

            logging.info(f"Dataset + optimiser/scheduler reset at Epoch: {current_epoch} and Batch: {current_batch}).")

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
                filename="{epoch:02d}", # epoch00/, epoch01/, …
                every_n_epochs=1,
                save_top_k=-1, # keep all epochs
                save_on_train_epoch_end=True, # Save at end of training epoch, not validation
                **kwargs,
            )
        else:
            super().__init__(
                dirpath=ckpt_root,
                filename="{step:04d}", # step0001/, …
                monitor="val_loss",
                mode="min",
                save_top_k=1, # keep only the best
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

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
from src.train.utils import is_rank_zero

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
            # Setup special tokens
            self._setup_special_tokens()
        else:
            self.model = AutoModelForCausalLM.from_pretrained(self.config["model"]["base_model_name_or_path"])
        
        # Store training config for optimizer setup
        self.learning_rate = float(self.config["training"]["learning_rate"])
        self.weight_decay = self.config["training"]["weight_decay"]
        self.lr_scheduler_step_size = self.config["training"]["lr_scheduler_step_size"]
        self.lr_scheduler_gamma = self.config["training"]["lr_scheduler_gamma"]

    def _setup_special_tokens(self):
        """Setup special tokens for latent reasoning."""
        START = "<|start-latent|>"
        LAT = "<|latent|>"
        END = "<|end-latent|>"
        new_specials = [START, LAT, END]

        # Only add / resize / init if it is not the first training run
        if not all(tok in self.tokenizer.get_vocab() for tok in new_specials):
            # Add them to the tokenizer's vocab
            self.tokenizer.add_tokens(new_specials)
            self.model.resize_token_embeddings(len(self.tokenizer))  # expand model embeddings

            # Save them as attributes for easy access
            self.tokenizer.start_latent_token = START
            self.tokenizer.latent_token = LAT
            self.tokenizer.end_latent_token = END

            self.tokenizer.start_latent_token_id = self.tokenizer.convert_tokens_to_ids(START)
            self.tokenizer.latent_token_id = self.tokenizer.convert_tokens_to_ids(LAT)
            self.tokenizer.end_latent_token_id = self.tokenizer.convert_tokens_to_ids(END)

            # mirror them on your model
            self.model.start_latent_token_id = self.tokenizer.start_latent_token_id
            self.model.latent_token_id = self.tokenizer.latent_token_id
            self.model.end_latent_token_id = self.tokenizer.end_latent_token_id
            
            # Get the embedding layer correctly
            embedding_layer = self.model.get_input_embeddings()
            
            # Init the new latent tokens
            vocab = self.tokenizer.get_vocab()
            # Use torch.no_grad() to safely modify the weights
            with torch.no_grad():
                # copy existing tokens
                embedding_layer.weight[self.model.start_latent_token_id] = embedding_layer.weight[vocab["."]].clone()
                embedding_layer.weight[self.model.end_latent_token_id] = embedding_layer.weight[vocab["."]].clone()
        else:
            # tokens already there – still handy to have the ids on the objects
            self.tokenizer.start_latent_token_id = self.tokenizer.convert_tokens_to_ids(START)
            self.tokenizer.latent_token_id = self.tokenizer.convert_tokens_to_ids(LAT)
            self.tokenizer.end_latent_token_id = self.tokenizer.convert_tokens_to_ids(END)
            self.model.start_latent_token_id = self.tokenizer.start_latent_token_id
            self.model.latent_token_id = self.tokenizer.latent_token_id
            self.model.end_latent_token_id = self.tokenizer.end_latent_token_id
            logging.info("Special tokens already present – skipping re-initialisation.")
    
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
        
    def setup(self, stage: str):
        """Setup datasets."""
        # Prepare the dataset
        # If latent reasoning is enabled, prepare the dataset with latent tokens
        if self.is_latent_reasoner:
            max_num_latent_steps = self.config["training"]["max_num_latent_steps"]
        else:
            # For standard SFT, no latent steps are used
            max_num_latent_steps = None

        data = prepare_dataset_latent_sft(dataset_name=self.config["dataset"]["name"], 
                                          tokenizer=self.tokenizer, 
                                          seed=self.config["training"]["seed"],
                                          max_num_latent_steps=max_num_latent_steps)
        
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

            print("\n=== SFT Data Check ===")
            print("Prompt:\n", decoded_prompt)
            # if you store the answer/label under another key, adjust here
            if "labels" in sample:
                print("\nLabel IDs:", sample["labels"])
            print("====================================\n")        

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
            if self.is_latent_reasoner:
                cut_idx = full_ids.index(self.tokenizer.start_latent_token_id) # first <|start-latent|>
            else: 
                return # don't generate samples for non-latent reasoner models since we can't detect the answer start for now

            prompt_ids = full_ids[:cut_idx]

            # back to tensor for generation
            input_ids = torch.tensor(prompt_ids, dtype=torch.long, device=pl_module.device).unsqueeze(0)
            attention_mask = torch.ones_like(input_ids)

            generated_ids = pl_module.model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                num_latent_steps=10,
                max_new_tokens=256,
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

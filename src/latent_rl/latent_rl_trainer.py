import torch
import math
import wandb
import logging
import os
from tqdm import tqdm


class LatentRLTrainer():

    def __init__(
        self, tokenizer, model, value_model, train_loader,
        reward_funcs: list, num_latent_steps: int,
        generation_config, args, device: str
    ) -> None:
        self.model = model
        self.value_model = value_model
        self.train_loader = train_loader
        self.reward_funcs = reward_funcs
        self.reward_func_names = []
        for i, _ in enumerate(reward_funcs):
            self.reward_func_names.append(reward_funcs[i].__name__)
        self.processing_class = tokenizer
        self.num_latent_steps = num_latent_steps
        self.generation_config = generation_config
        self.args = args
        self.device = device
        
        # Initialize missing attributes
        self.global_step = 0
        self.training_metrics = []
        
        # Initialize dual optimizers
        self.rl_optimizer, self.value_head_optimizer = self._setup_optimizer()


    def _setup_optimizer(self):
        """Setup optimizers for training - separate optimizers for different components"""
        lr = self.args.get("learning_rate")
        value_head_lr = self.args.get("value_head_learning_rate")
        weight_decay = self.args.get("weight_decay", 0.01)
        
        # Optimizer for latent reasoner (RL training)
        # Use value_model parameters EXCEPT the value_head since model and value_model share/tie params
        value_head_param_ids = {id(p) for p in self.value_model.value_head.parameters()}
        base_value_model_params = [
            p for p in self.value_model.parameters() if id(p) not in value_head_param_ids
        ]
        rl_optimizer = torch.optim.AdamW(
            base_value_model_params,
            lr=lr,
            weight_decay=weight_decay
        )
        
        # Optimizer for value head (supervised training)
        value_head_optimizer = torch.optim.AdamW(
            self.value_model.value_head.parameters(), 
            lr=value_head_lr, 
            weight_decay=weight_decay
        )
        
        logging.info(f"Optimizers with LM LR: {lr} and Value Head LR: {value_head_lr} are initialized.")

        return rl_optimizer, value_head_optimizer


    def _create_scheduler(self, optimizer, total_update_steps: int):
        """Cosine decay with warmup and min_lr_rate tail."""
        warmup_steps = int(self.args.get("warmup_steps", 0))
        lr_sched_kwargs = self.args.get("lr_scheduler_kwargs", {}) or {}
        min_lr_rate = float(lr_sched_kwargs.get("min_lr_rate", 0.0))

        total_update_steps = max(int(total_update_steps), 1)
        warmup_steps = max(int(warmup_steps), 0)
        non_warmup_steps = max(total_update_steps - warmup_steps, 1)

        def lr_lambda(current_step: int):
            if warmup_steps > 0 and current_step < warmup_steps:
                return float(current_step) / float(max(1, warmup_steps))
            progress = float(current_step - warmup_steps) / float(non_warmup_steps)
            progress = min(max(progress, 0.0), 1.0)
            cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
            return min_lr_rate + (1.0 - min_lr_rate) * cosine

        return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


    def _generate_and_score_completions(self, inputs) -> dict:
        #### ASSUMPTION: We assume all completions in the batch have the same number of latent steps    ####
        #### and they are all at the beginning of the completion sequence                               ####
        #### If this doesn't hold, change this implementation                                           ####

        device = self.device

        prompts = inputs["prompt"]
        prompt_inputs = self.processing_class(
            text=prompts, return_tensors="pt", padding=True, padding_side="left", add_special_tokens=False
        )
        prompt_ids, prompt_mask = prompt_inputs["input_ids"].to(device), prompt_inputs["attention_mask"].to(device)

        # Generate completions using the model
        # This custom generate method from Latent reasoner returns:
        # 1) the completion token IDs for latent steps + language steps, no prompt
        # 2) and the embeddings for the prompt + latent steps + language steps
        completion_ids, prompt_completion_embeds = self.model.generate(
            prompt_ids, attention_mask=prompt_mask, 
            num_latent_steps=self.num_latent_steps,
            max_new_tokens=self.generation_config['max_completion_length'],
            temperature=self.generation_config['temperature'] if self.generation_config['temperature'] > 0 else None,
            do_sample=True if self.generation_config['temperature'] > 0 else False,
            pad_token_id=self.processing_class.pad_token_id,
        )
        prompt_completion_embeds = prompt_completion_embeds.detach()

        # Compute prompt length and extract completion ids
        prompt_length = prompt_ids.size(1)
        prompt_embeds = prompt_completion_embeds[:, :prompt_length]
        completion_embeds = prompt_completion_embeds[:, prompt_length:]

        # Mask everything after the first EOS token
        is_eos = completion_ids == self.processing_class.eos_token_id
        eos_idx = torch.full((is_eos.size(0),), is_eos.size(1), dtype=torch.long, device=device)
        eos_idx[is_eos.any(dim=1)] = is_eos.int().argmax(dim=1)[is_eos.any(dim=1)]
        sequence_indices = torch.arange(is_eos.size(1), device=device).expand(is_eos.size(0), -1)
        completion_mask = (sequence_indices <= eos_idx.unsqueeze(1)).int()

        # Create another mask for latent steps since only they will be used for gradient calculation
        is_start = completion_ids == self.processing_class.start_latent_token_id
        is_end = completion_ids == self.processing_class.end_latent_token_id
        start_seen = is_start.cumsum(dim=1)
        end_seen   = is_end.cumsum(dim=1)
        inside_block = (start_seen >= 1) & (end_seen == 0)
        inside_block &= ~is_start
        latent_mask_completion = ((completion_ids == self.processing_class.latent_token_id) & inside_block).long()

        # ASSUMPTION: We assume all completions in the batch have the same number of latent steps
        assert latent_mask_completion.sum(dim=1).unique().numel() == 1, "All completions in batch should have same number of latent steps"

        # Add the zeros in the prompt and concat with latent_mask_completions to create is_latent_mask
        latent_mask = torch.cat([torch.zeros_like(prompt_ids), latent_mask_completion], dim=1)

        # Convert tensor to a list of lists of token IDs. This will be passed to the reward function, avoiding the need
        # to re-tokenize completions if the reward is computed from tokens.
        completion_ids_list = [
            [id.item() for id, m in zip(row, mask_row) if m] for row, mask_row in zip(completion_ids, completion_mask)
        ]

        # Decode the generated completions
        completions_text = self.processing_class.batch_decode(completion_ids, skip_special_tokens=True)

        rewards = {}
        for _, (reward_func, reward_func_name) in enumerate(
            zip(self.reward_funcs, self.reward_func_names)
        ):
                output_reward_func = reward_func(
                    prompts=prompts, completions=completions_text, completion_ids=completion_ids_list, answer=inputs['answer']
                )
                # Convert None values to NaN
                output_reward_func = [reward if reward is not None else torch.nan for reward in output_reward_func]
                rewards[reward_func_name] = torch.tensor(output_reward_func, dtype=torch.float32, device=device)

        return {
            "prompt_ids": prompt_ids,
            "completion_ids": completion_ids,
            "prompt_completion_embeds": prompt_completion_embeds,
            "latent_mask": latent_mask,
            "rewards": rewards
        }


    def train_step(self, inputs):
        # Generate response with LatR model - single forward pass
        generation_results = self._generate_and_score_completions(inputs)
        
        # Single forward pass through the Value Model
        values_logits = self.value_model(inputs_embeds=generation_results["prompt_completion_embeds"])
        values = torch.sigmoid(values_logits)

        # Extract latent mask and rewards
        latent_mask = generation_results["latent_mask"]  # Shape: [batch_size, seq_len]
        accuracy_rewards = generation_results["rewards"]["accuracy_reward"]  # [batch_size]
        num_latent_tokens = latent_mask.sum()
        
        # === RL Loss (for latent reasoner) ===
        latent_rewards = values * latent_mask.float()
        if num_latent_tokens > 0:
            avg_latent_reward = latent_rewards.sum() / num_latent_tokens
            rl_loss = -avg_latent_reward
        else:
            rl_loss = torch.tensor(0.0, device=self.device, requires_grad=True)
        
        # === Value Head Loss (supervised) ===
        # Prepare targets for value head
        expanded_targets = accuracy_rewards.unsqueeze(1).expand(-1, values_logits.size(1))
        masked_targets = expanded_targets * latent_mask.float()
        masked_logits = values_logits * latent_mask.float()
        
        if num_latent_tokens > 0:
            flat_logits = masked_logits[latent_mask.bool()]
            flat_targets = masked_targets[latent_mask.bool()]
            value_head_loss = torch.nn.functional.binary_cross_entropy_with_logits(
                flat_logits, flat_targets, 
                pos_weight=torch.tensor([self.args.get("value_head_pos_weight")], dtype=torch.float32).to(self.device)
            )
        else:
            value_head_loss = torch.tensor(0.0, device=self.device, requires_grad=True)
        
        # === Backward passes ===
        # 1. RL loss backward (updates latent reasoner)
        rl_loss.backward(retain_graph=True)
        
        # 2. Value head loss backward (updates value head)
        value_head_loss.backward()
        
        return {
            "loss": rl_loss.item(),
            "value_head_loss": value_head_loss.item(),
            "num_latent_tokens": num_latent_tokens.item() if num_latent_tokens > 0 else 0,
            "generation_results": generation_results
        }

    def train(self):
        """Complete training function with dual optimizers"""
        # Training configuration
        epochs = self.args.get("num_train_epochs", 1)
        gradient_accumulation_steps = self.args.get("gradient_accumulation_steps", 1)
        max_grad_norm = self.args.get("max_grad_norm", 1.0)
        
        logging.info(f"Starting training for {epochs} epochs")
        logging.info(f"Total batches per epoch: {len(self.train_loader)}")
        
        # Set models to training mode
        self.value_model.train()

        # Number of optimizer update steps per epoch (account for gradient accumulation)
        steps_per_epoch = (len(self.train_loader) + gradient_accumulation_steps - 1) // gradient_accumulation_steps
        total_update_steps = max(epochs * steps_per_epoch, 1)

        # Setup LR schedulers after we know total update steps
        self.rl_scheduler = self._create_scheduler(self.rl_optimizer, total_update_steps)
        
        for epoch in range(epochs):
            logging.info(f"Starting epoch {epoch + 1}/{epochs}")
            
            # Create progress bar
            pbar = tqdm(self.train_loader, desc=f"Epoch {epoch + 1}")
            
            epoch_losses = []
            epoch_value_head_losses = []
            accumulated_loss = 0.0
            accumulated_value_head_loss = 0.0
            # Accumulators for averaging metrics across gradient accumulation window
            ga_loss_sum = 0.0
            ga_vh_loss_sum = 0.0
            ga_num_latent_tokens_sum = 0.0
            ga_micro_count = 0
            ga_rewards_accumulator = {}
            ga_last_generation_results = None
            
            for batch_idx, batch in enumerate(pbar):
                # Forward pass and compute losses
                step_results = self.train_step(batch)
                
                # Accumulate losses for logging
                accumulated_loss += step_results["loss"]
                accumulated_value_head_loss += step_results["value_head_loss"]
                epoch_losses.append(step_results["loss"])
                epoch_value_head_losses.append(step_results["value_head_loss"])
                
                # Accumulate per-micro-step metrics for GA window
                ga_loss_sum += step_results["loss"]
                ga_vh_loss_sum += step_results["value_head_loss"]
                ga_num_latent_tokens_sum += step_results["num_latent_tokens"]
                ga_micro_count += 1
                ga_last_generation_results = step_results["generation_results"]
                # Aggregate rewards across micro-steps for accurate stats over the GA window
                for r_name, r_tensor in step_results["generation_results"]["rewards"].items():
                    ga_rewards_accumulator.setdefault(r_name, []).append(r_tensor)
                
                # Gradient accumulation
                if (batch_idx + 1) % gradient_accumulation_steps == 0:                        
                    # Clip gradients for both optimizers
                    if max_grad_norm > 0:
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_grad_norm)
                        torch.nn.utils.clip_grad_norm_(self.value_model.value_head.parameters(), max_grad_norm)
                    
                    # Optimizer steps
                    self.rl_optimizer.step()
                    self.value_head_optimizer.step()
                    self.rl_optimizer.zero_grad()
                    self.value_head_optimizer.zero_grad()
                    
                    # Update global step counter
                    self.global_step += 1

                    # Scheduler steps (after optimizer steps)
                    if hasattr(self, "rl_scheduler") and self.rl_scheduler is not None:
                        self.rl_scheduler.step()
                    if hasattr(self, "value_head_scheduler") and self.value_head_scheduler is not None:
                        self.value_head_scheduler.step()
                    
                    # Calculate and log averaged metrics across the GA window
                    aggregated_generation_results = ga_last_generation_results
                    aggregated_generation_results["rewards"] = {
                        r_name: torch.cat(r_list, dim=0) for r_name, r_list in ga_rewards_accumulator.items()
                    }

                    averaged_step_metrics = {
                        "loss": ga_loss_sum / ga_micro_count,
                        "value_head_loss": ga_vh_loss_sum / ga_micro_count,
                        "num_latent_tokens": ga_num_latent_tokens_sum / ga_micro_count,
                    }

                    self._calculate_and_log_metrics(
                        aggregated_generation_results,
                        averaged_step_metrics
                    )
                    
                    # Reset GA accumulators
                    ga_loss_sum = 0.0
                    ga_vh_loss_sum = 0.0
                    ga_num_latent_tokens_sum = 0.0
                    ga_micro_count = 0
                    ga_rewards_accumulator = {}
                    ga_last_generation_results = None
                    
                    # Update progress bar
                    avg_loss = accumulated_loss / gradient_accumulation_steps
                    avg_value_head_loss = accumulated_value_head_loss / gradient_accumulation_steps
                    pbar.set_postfix({
                        'rl_loss': f'{avg_loss:.4f}',
                        'vh_loss': f'{avg_value_head_loss:.4f}',
                        'step': self.global_step
                    })
                    accumulated_loss = 0.0
                    accumulated_value_head_loss = 0.0
            
            # Log epoch statistics
            if epoch_losses:
                avg_epoch_loss = sum(epoch_losses) / len(epoch_losses)
                avg_epoch_value_head_loss = sum(epoch_value_head_losses) / len(epoch_value_head_losses)
                logging.info(f"Epoch {epoch + 1} completed. Average RL loss: {avg_epoch_loss:.4f}, Average value head loss: {avg_epoch_value_head_loss:.4f}")
            
            # Save checkpoint at end of epoch
            self._save_checkpoint(is_epoch_end=True, epoch=epoch)
        
        logging.info("Training completed!")
        wandb.finish()
        return self.training_metrics
    
    def _save_checkpoint(self, is_epoch_end=False, epoch=None):
        """Save model checkpoint"""
        save_dir = self.args.get("output_dir", "./checkpoints")
        
        # Create separate directories for model and value model
        model_dir = f"{save_dir}/model"
        # value_model_dir = f"{save_dir}/value_model"
        
        # Ensure directories exist
        os.makedirs(model_dir, exist_ok=True)
        # os.makedirs(value_model_dir, exist_ok=True)
        
        if is_epoch_end:
            model_path = f"{model_dir}/checkpoint_epoch_{epoch}"
            # value_model_path = f"{value_model_dir}/checkpoint_epoch_{epoch}"
        else:
            model_path = f"{model_dir}/checkpoint_step_{self.global_step}"
            # value_model_path = f"{value_model_dir}/checkpoint_step_{self.global_step}"
        
        try:
            # Save main model in Hugging Face format
            self.model.save_pretrained(model_path)
            logging.info(f"Model saved in Hugging Face format to {model_path}")
            
            # # Save value model in Hugging Face format
            # self.value_model.save_pretrained(value_model_path)
            # logging.info(f"Value model saved in Hugging Face format to {value_model_path}")
            
        except Exception as e:
            logging.error(f"Failed to save checkpoint: {str(e)}")

    def _calculate_and_log_metrics(self, generation_results, step_metrics):
        """
        Calculate and log training metrics including:
        - rewards from different reward functions
        - estimated rewards by the value model
        - responses generated by LatR
        - create a table for those stuff containing prompt, response, estimated reward, real reward
        """
        rewards = generation_results["rewards"]
        prompt_ids = generation_results["prompt_ids"]
        completion_ids = generation_results["completion_ids"]
        
        # Decode prompts and completions for logging
        prompts = self.processing_class.batch_decode(prompt_ids, skip_special_tokens=True)
        completions = self.processing_class.batch_decode(completion_ids, skip_special_tokens=True)
        
        # Log reward statistics
        reward_stats = {}
        for reward_name, reward_tensor in rewards.items():
            valid_rewards = reward_tensor[~torch.isnan(reward_tensor)]
            if len(valid_rewards) > 0:
                reward_stats[f"{reward_name}_mean"] = valid_rewards.mean().item()
                reward_stats[f"{reward_name}_std"] = valid_rewards.std().item()
                reward_stats[f"{reward_name}_min"] = valid_rewards.min().item()
                reward_stats[f"{reward_name}_max"] = valid_rewards.max().item()
        
        # Combine all metrics
        metrics = {
            "step": self.global_step,
            "loss": step_metrics["loss"],
            "value_head_loss": step_metrics["value_head_loss"],
            "num_latent_tokens": step_metrics["num_latent_tokens"],
            **reward_stats
        }
        
        self.training_metrics.append(metrics)
        
        # Save examples for inspection (every 10 steps)
        if self.global_step % self.args.get("response_logging_steps", 5) == 0:
            # Initialize examples table data if not exists
            if not hasattr(self, 'examples_table_data'):
                self.examples_table_data = []
                
            # Determine the actual batch size to avoid index errors
            batch_size = len(prompts)
            num_examples = min(batch_size, 5)  # Take up to 5 examples or batch size, whichever is smaller
            
            reward_names = list(rewards.keys())
            
            # Add new examples to persistent table
            for i in range(num_examples):
                # Create a row with step, prompt, completion, and all reward values
                row = [self.global_step, prompts[i], completions[i]]
                
                # Add reward values for this example, handling potential tensor size mismatches
                for reward_name in reward_names:
                    if i < len(rewards[reward_name]):
                        reward_val = rewards[reward_name][i].item()
                        # Handle NaN values
                        row.append(reward_val if not torch.isnan(torch.tensor(reward_val)) else "NaN")
                    else:
                        row.append("N/A")
                
                self.examples_table_data.append(row)
            
            # Create table with proper column names
            columns = ["step", "prompt", "completion"] + reward_names
            
            # Log the cumulative table
            wandb.log({
                "examples": wandb.Table(columns=columns, data=self.examples_table_data)
            }, step=self.global_step)
        
        # Log scalar metrics
        wandb.log(metrics, step=self.global_step)

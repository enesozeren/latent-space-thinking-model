import torch
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
        
        self.optimizer = self._setup_optimizer()


    def _setup_optimizer(self):
        """Setup optimizer for training"""
        lr = self.args.get("learning_rate", 5e-5)
        weight_decay = self.args.get("weight_decay", 0.01)
        
        optimizer = torch.optim.AdamW(
            self.value_model.parameters(), # since the value model and model parameters are tied
            lr=lr, 
            weight_decay=weight_decay
        )
        
        return optimizer


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
        # generate a response with LatR model
        ## this returns prompt and completion token ids and embeddings, rewards, mask (latent steps: 1, others 0)
        generation_results = self._generate_and_score_completions(inputs)
        
        # freeze the value model head
        self.value_model.freeze_value_head()        
        # pass all the prompt+completion embeddings through the Value Model (tied with LatR model + value model head)
        values_logits = self.value_model(inputs_embeds=generation_results["prompt_completion_embeds"])
        # pass the values_logits through sigmoid
        values = torch.sigmoid(values_logits)

        # Extract latent mask and create target rewards
        latent_mask = generation_results["latent_mask"]  # Shape: [batch_size, seq_len]
        
        # Use value model predictions as rewards for latent steps (RL objective)
        # The value model evaluates how "good" each latent step embedding is]
        
        # Apply latent mask to extract rewards only for latent steps
        latent_rewards = values * latent_mask.float()  # Zero out non-latent steps
        
        # Calculate RL loss: maximize the value model's predictions for latent steps
        # We use negative values as loss (since we want to maximize rewards)
        num_latent_tokens = latent_mask.sum()
        if num_latent_tokens > 0:
            # Average reward over latent steps, then negate for loss
            avg_latent_reward = latent_rewards.sum() / num_latent_tokens
            loss = -avg_latent_reward  # Maximize reward = minimize negative reward
        else:
            loss = torch.tensor(0.0, device=self.device, requires_grad=True)
        
        # backprop through the Value Model (so the LatR model is also updated)
        loss.backward()
        
        return {
            "loss": loss.item(),
            "num_latent_tokens": num_latent_tokens.item() if num_latent_tokens > 0 else 0,
            "generation_results": generation_results
        }

    def train(self):
        """Complete training function with proper optimization loop"""
        # Training configuration
        epochs = self.args.get("num_train_epochs", 1)
        gradient_accumulation_steps = self.args.get("gradient_accumulation_steps", 1)
        max_grad_norm = self.args.get("max_grad_norm", 1.0)
        
        logging.info(f"Starting training for {epochs} epochs")
        logging.info(f"Total batches per epoch: {len(self.train_loader)}")
        
        # Set models to training mode
        self.value_model.train()

        total_steps = epochs * len(self.train_loader)
        
        for epoch in range(epochs):
            logging.info(f"Starting epoch {epoch + 1}/{epochs}")
            
            # Create progress bar
            pbar = tqdm(self.train_loader, desc=f"Epoch {epoch + 1}")
            
            epoch_losses = []
            accumulated_loss = 0.0
            
            for batch_idx, batch in enumerate(pbar):
                try:
                    # Forward pass and compute loss
                    step_results = self.train_step(batch)
                    
                    # Accumulate loss for logging
                    accumulated_loss += step_results["loss"]
                    epoch_losses.append(step_results["loss"])
                    
                    # Gradient accumulation
                    if (batch_idx + 1) % gradient_accumulation_steps == 0:                        
                        # Clip gradients
                        if max_grad_norm > 0:
                            torch.nn.utils.clip_grad_norm_(
                                list(self.model.parameters()) + list(self.value_model.parameters()),
                                max_grad_norm
                            )
                        
                        # Optimizer step
                        self.optimizer.step()
                        self.optimizer.zero_grad()
                        
                        # Update global step counter
                        self.global_step += 1
                        
                        # Calculate and log metrics
                        self._calculate_and_log_metrics(
                            step_results["generation_results"], 
                            step_results
                        )
                        
                        # Update progress bar
                        avg_loss = accumulated_loss / gradient_accumulation_steps
                        pbar.set_postfix({
                            'loss': f'{avg_loss:.4f}',
                            'step': self.global_step
                        })
                        accumulated_loss = 0.0
                        
                except Exception as e:
                    logging.error(f"Error in batch {batch_idx}: {str(e)}")
                    # Continue training with next batch
                    continue
            
            # Log epoch statistics
            if epoch_losses:
                avg_epoch_loss = sum(epoch_losses) / len(epoch_losses)
                logging.info(f"Epoch {epoch + 1} completed. Average loss: {avg_epoch_loss:.4f}")
            
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
            "num_latent_tokens": step_metrics["num_latent_tokens"],
            **reward_stats
        }
        
        self.training_metrics.append(metrics)
        
        # Save examples for inspection (every 10 steps)
        if self.global_step % 10 == 0:
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

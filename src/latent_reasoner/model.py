import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

class LatentReasoner:
    def __init__(self, model_name: str, num_latent_steps: int = 5):
        # Supported model check
        if model_name not in ["Qwen/Qwen2.5-1.5B", "Qwen/Qwen2.5-3B"]:
            raise ValueError(f"Model {model_name} is not supported.")

        self.num_latent_steps = num_latent_steps
        # Load model and tokenizer
        self.model_name = model_name
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        # Register special tokens: latent start/end + display-only latent marker
        self.tokenizer.add_special_tokens({
            "additional_special_tokens": [
                "<|start-latent|>",
                "<|end-latent|>",
                "<|latent|>"
            ]
        })
        # Resize the model's token embeddings to accommodate new tokens
        self.model.resize_token_embeddings(len(self.tokenizer))
        # Get the embedding layer
        self.embeding_layer = self.model.get_input_embeddings()
        # Init embeddings for new latent tokens
        self._init_latent_tokens()

    def _init_latent_tokens(self):
        with torch.no_grad():
            vocab = self.tokenizer.get_vocab()
            sid = vocab.get("<|start-latent|>")
            eid = vocab.get("<|end-latent|>")
            lid = vocab.get("<|latent|>")
            # copy from similar tokens
            if "<|im_start|>" in vocab and sid is not None:
                self.embeding_layer.weight[sid] = self.embeding_layer.weight[vocab["<|im_start|>"]]
            if "<|im_end|>" in vocab and eid is not None:
                self.embeding_layer.weight[eid] = self.embeding_layer.weight[vocab["<|im_end|>"]]
            # init latent token with 0s since it won't be feed to the model ever
            if lid is not None:
                self.embeding_layer.weight[lid] = torch.zeros_like(self.embeding_layer.weight[0])
    
    def train(self):
        self.model.train()

    def eval(self):
        self.model.eval()
        
    def forward(self, input_embeddings, attention_mask):
        
        output = self.model(
            inputs_embeds=input_embeddings,
            attention_mask=attention_mask,
            output_hidden_states=True,
            return_dict=True
        )

        # Get the last hidden state and logits
        last_hidden_states = output.hidden_states[-1]
        logits = output.logits

        return last_hidden_states, logits

    def generate(
        self,
        inputs,
        attention_mask,
        generation_config = None,
        prompts: list = None, # Currently not supporting VLLM (we dont use the prompts input)
        max_new_tokens: int = 20,
        temperature: float = 1.0,
        top_k: int = 50,
        top_p: float = 1.0,
        device: str = None
    ) -> str:
        batch_size = inputs.size(0)

        # Special token ids
        start_latent_id = self.tokenizer.get_vocab()["<|start-latent|>"]
        end_latent_id = self.tokenizer.get_vocab()["<|end-latent|>"]
        latent_id = self.tokenizer.get_vocab()["<|latent|>"]
        padding_id = self.tokenizer.pad_token_id

        # Get input embeddings
        inputs_embeds = self.embeding_layer(inputs)
        finished = torch.zeros(batch_size, dtype=torch.bool, device=device)
        
        generated_seqs: list[list[int]] = [[] for _ in range(batch_size)]

        def _append_to_all(seqs, tid):
            for s in seqs:
                s.append(tid)

        # Latent space generation
        for _ in range(self.num_latent_steps):
            last_hidden_states, _ = self.forward(inputs_embeds, attention_mask)
            _append_to_all(generated_seqs, latent_id)
            # Append last latent embedding
            inputs_embeds = torch.cat([inputs_embeds, last_hidden_states[:, -1:, :]], dim=1)
            attention_mask = torch.cat([attention_mask, torch.ones((batch_size, 1), dtype=attention_mask.dtype, device=attention_mask.device)], dim=1)
        
        # Append end-of-latent marker
        end_latent_emb = self.embeding_layer(torch.full((batch_size, 1), end_latent_id, device=device, dtype=torch.long))
        inputs_embeds = torch.cat([inputs_embeds, end_latent_emb], dim=1)
        attention_mask = torch.cat([attention_mask, torch.ones((batch_size, 1), dtype=attention_mask.dtype, device=attention_mask.device)], dim=1)
        # Add end latent token to generated sequences
        _append_to_all(generated_seqs, end_latent_id)

        # Language space generation (greedy decoding)
        for _ in range(max_new_tokens):
            last_hidden_states, logits = self.forward(inputs_embeds, attention_mask)
            # Get logits for the last position
            next_logits = logits[:, -1, :] / temperature
            next_tokens = torch.argmax(next_logits, dim=-1)

            # update per-sequence state
            for i, tok in enumerate(next_tokens.tolist()):
                if finished[i]:
                    continue
                generated_seqs[i].append(tok)
                if tok == self.tokenizer.eos_token_id:
                    finished[i] = True

            # break early if everybody is done
            if torch.all(finished):
                break

            # feed placeholders for finished sequences
            next_tokens_for_embed = next_tokens.masked_fill(finished, padding_id).unsqueeze(-1)
            next_emb = self.embeding_layer(next_tokens_for_embed)

            inputs_embeds = torch.cat([inputs_embeds, next_emb], dim=1)
            attention_mask = torch.cat(
                [attention_mask,
                (~finished).long().unsqueeze(-1)],   # 1 for active seqs, 0 otherwise
                dim=1,
            )

        # Generated tokens
        max_len = max(len(s) for s in generated_seqs)
        generated = torch.full(
            (batch_size, max_len),
            padding_id,
            dtype=torch.long,
            device=device,
        )
        for i, seq in enumerate(generated_seqs):
            generated[i, : len(seq)] = torch.tensor(seq, device=device, dtype=torch.long)

        return generated

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    lr = LatentReasoner(model_name="Qwen/Qwen2.5-1.5B", num_latent_steps=5)
    
    # Define two prompts for batch processing
    prompts = [
        "User: What is the capital of Germany?",
        "User: Can you explain quantum computing in simple terms?"
    ]
    
    # Process both prompts
    prompts = [p + "\nAssistant: " + "<|start-latent|>" for p in prompts]
    
    # Tokenize batch of prompts
    tokenized = lr.tokenizer(prompts, return_tensors="pt", padding=True)
    prompt_ids = tokenized["input_ids"].to(device)
    attention_mask = tokenized["attention_mask"].to(device)
    
    # Generate responses for the batch
    responses = lr.generate(
        prompt_ids,
        attention_mask=attention_mask,
        max_new_tokens=20,
        generation_config=None,
        device=device,
    )

    # Decode the batch of generated responses
    response_texts = lr.tokenizer.batch_decode(responses, skip_special_tokens=False)
    
    # Print each prompt and its corresponding response
    for i, (prompt, response) in enumerate(zip(prompts, response_texts)):
        print(f"Prompt {i+1}: {prompt!r}")
        print(f"Response {i+1}: {response!r}")
        print("-" * 50)

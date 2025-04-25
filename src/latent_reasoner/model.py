import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Custom filtering function compatible with transformers>=4.41
def top_k_top_p_filtering(
    logits: torch.Tensor,
    top_k: int = 0,
    top_p: float = 1.0,
    filter_value: float = -float("Inf")
) -> torch.Tensor:
    """
    Filter a distribution of logits using top-k and/or nucleus (top-p) filtering
    Args:
        logits: logits distribution shape (..., vocab_size)
        top_k > 0: keep only top_k tokens with highest probability.
        top_p < 1.0: keep the top tokens with cumulative probability >= top_p.
    """
    # Top-K filtering
    if top_k > 0:
        # Determine the cutoff value
        top_k = min(max(top_k, 1), logits.size(-1))
        # Remove tokens with a probability less than the top-k threshold
        values_to_keep = torch.topk(logits, top_k, dim=-1).values[..., -1, None]
        logits = torch.where(
            logits < values_to_keep,
            torch.full_like(logits, filter_value),
            logits
        )
    # Top-P (nucleus) filtering
    if top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
        cumulative_probs = torch.softmax(sorted_logits, dim=-1).cumsum(dim=-1)
        # Remove tokens with cumulative probability above threshold
        sorted_indices_to_remove = cumulative_probs > top_p
        # Always keep at least the first token
        sorted_indices_to_remove[..., 0] = False
        # Scatter back to original indexing
        indices_to_remove = sorted_indices_to_remove.scatter(
            -1, sorted_indices, sorted_indices_to_remove
        )
        logits = logits.masked_fill(indices_to_remove, filter_value)
    return logits

class LatentReasoner:
    def __init__(self, model_name: str):
        # Supported model check
        if model_name not in ["Qwen/Qwen2.5-1.5B-Instruct"]:
            raise ValueError(f"Model {model_name} is not supported.")

        # Load tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        # Register special tokens: latent start/end + display-only latent marker
        self.tokenizer.add_special_tokens({
            "additional_special_tokens": [
                "<|start-latent|>",
                "<|end-latent|>",
                "<|latent|>"
            ]
        })
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        self.model.resize_token_embeddings(len(self.tokenizer))

        # Init embeddings for new latent tokens
        self._init_latent_tokens()
        # Binary stop classifier
        self._init_end_latent_classifier()

        # Ensure hidden states are returned
        self.model.config.output_hidden_states = True
        # Default inference mode
        self.model.eval()

    def _init_latent_tokens(self):
        embeddings = self.model.get_input_embeddings()
        with torch.no_grad():
            vocab = self.tokenizer.get_vocab()
            sid = vocab.get("<|start-latent|>")
            eid = vocab.get("<|end-latent|>")
            # copy from similar tokens when available
            if "<|im_start|>" in vocab and sid is not None:
                embeddings.weight[sid] = embeddings.weight[vocab["<|im_start|>"]]
            if "<|im_end|>" in vocab and eid is not None:
                embeddings.weight[eid] = embeddings.weight[vocab["<|im_end|>"]]

    def _init_end_latent_classifier(self):
        self.end_latent_classifier = torch.nn.Linear(
            self.model.config.hidden_size, 1
        )
        with torch.no_grad():
            self.end_latent_classifier.weight.normal_(
                mean=0.0, std=self.model.config.initializer_range
            )
            self.end_latent_classifier.bias.zero_()

    @torch.no_grad()
    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 100,
        latent_max_steps: int = 20,
        stop_threshold: float = 0.5,
        temperature: float = 1.0,
        top_k: int = 50,
        top_p: float = 1.0,
        device: str = None
    ) -> str:
        """
        Generate with latent reasoning and display markers.
        """
        device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(device)
        self.end_latent_classifier.to(device)

        # Prepare prompt + <|start-latent|>
        sid = self.tokenizer.convert_tokens_to_ids("<|start-latent|>")
        enc = self.tokenizer.encode(prompt + self.tokenizer.eos_token, return_tensors="pt").to(device)
        toks = torch.cat([enc, torch.tensor([[sid]], device=device)], dim=-1)

        # Encode up to latent start
        out = self.model(input_ids=toks, use_cache=True)
        past = out.past_key_values
        hidden = out.hidden_states[-1][..., -1, :]

        # Track latent steps
        lid = self.tokenizer.convert_tokens_to_ids("<|latent|>")
        latent_count = 0

        for _ in range(latent_max_steps):
            logit = self.end_latent_classifier(hidden)
            if torch.sigmoid(logit).item() > stop_threshold:
                break
            latent_count += 1
            out = self.model(
                inputs_embeds=hidden.unsqueeze(1), past_key_values=past, use_cache=True
            )
            past = out.past_key_values
            hidden = out.hidden_states[-1][..., -1, :]

        # Inject latent end
        eid = self.tokenizer.convert_tokens_to_ids("<|end-latent|>")
        emb_end = self.model.get_input_embeddings()(torch.tensor([[eid]], device=device))
        out = self.model(inputs_embeds=emb_end, past_key_values=past, use_cache=True)
        past = out.past_key_values

        # Decode sampling
        generated = []
        for _ in range(max_new_tokens):
            logits = out.logits[:, -1, :]
            logits = logits / temperature
            logits = top_k_top_p_filtering(logits, top_k=top_k, top_p=top_p)
            probs = torch.softmax(logits, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1)
            if next_id.item() == self.tokenizer.eos_token_id:
                break
            generated.append(next_id.item())
            out = self.model(input_ids=next_id, past_key_values=past, use_cache=True)
            past = out.past_key_values

        # Build display sequence
        markers = [lid] * latent_count
        all_ids = markers + generated
        return self.tokenizer.decode(all_ids, skip_special_tokens=False).strip()

# if __name__ == "__main__":
#     lr = LatentReasoner("Qwen/Qwen2.5-1.5B-Instruct")
#     print(lr.generate("What's 2+2?", temperature=0.7, top_k=50, top_p=1.0))

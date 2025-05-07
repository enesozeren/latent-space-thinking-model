import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

class LatentReasoner:
    def __init__(self, model_name: str):
        # Supported model check
        if model_name not in ["Qwen/Qwen2.5-1.5B", "Qwen/Qwen2.5-3B"]:
            raise ValueError(f"Model {model_name} is not supported.")

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
        # Binary stop classifier with sigmoid activation
        self.end_latent_classifier = torch.nn.Sequential(
            torch.nn.Linear(self.model.config.hidden_size, 1),
            torch.nn.Sigmoid()
        )

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

        # Get the last hidden state
        last_hidden_states = output.hidden_states[-1]

        return last_hidden_states

    @torch.no_grad()
    def generate(
        self,
        inputs,
        attention_mask,
        generation_config = None,
        prompts: list = None, # Currently not supporting VLLM (we dont use the prompts input)
        max_new_tokens: int = 100,
        latent_max_steps: int = 20,
        stop_threshold: float = 0.5,
        temperature: float = 1.0,
        top_k: int = 50,
        top_p: float = 1.0,
        device: str = None
    ) -> str:
        
        inputs_embeds = self.embeding_layer(inputs)
        last_hidden_states = self.forward(inputs_embeds, attention_mask)
        
        # Write it in loop to generate tokens

if __name__ == "__main__":
    lr = LatentReasoner("Qwen/Qwen2.5-1.5B")
    prompt = "What is 2+2?"
    # Get input ids and attention mask
    tokenized = lr.tokenizer(prompt, return_tensors="pt")
    prompt_ids = tokenized["input_ids"]
    attention_mask = tokenized["attention_mask"]
    # Generate a response
    response = lr.generate(
        prompt_ids,
        attention_mask=attention_mask,
        generation_config=None  # Placeholder for generation config
    )

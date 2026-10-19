import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)

DEFAULT_SYSTEM_PROMPT = "You are a tennis tactical analyst. Be precise, evidence-based, and concise."

class Model:
    def __init__(self, name="Qwen/Qwen3-4B", chat_history=[], system_prompt=DEFAULT_SYSTEM_PROMPT):
        self.tokenizer = AutoTokenizer.from_pretrained(name)
        self.model = AutoModelForCausalLM.from_pretrained(
            name,
            quantization_config=bnb_config,
            device_map="auto"
        )

        self.model.eval()
        self.chat_history = chat_history if len(chat_history) != 0 else [
            {
                "role": "system",
                "content": system_prompt
            }
        ]

    def set_chat_history(self, new_history):
        self.chat_history = new_history

    def get_response(self, prompt, max_new_tokens=512, thinking=False):
        self.chat_history += [{"role": "user", "content": prompt}]

        formatted_prompt = self.tokenizer.apply_chat_template(
            self.chat_history,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=thinking,
        )

        inputs = self.tokenzier(formatted_prompt, return_tensors="pt").to(self.model.device)

        with torch.inference_mode():
            output = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id
            )

        new_tokens = output[0, inputs['input_ids'].shape[1]:]

        return self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


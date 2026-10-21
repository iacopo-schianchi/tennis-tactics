import gc
import re

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)

THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)


def strip_think(text):
    text = THINK_RE.sub("", text)
    if "<think>" in text:
        text = text.split("<think>")[0]
    return text.strip()


class Model:
    _instances = {}

    @classmethod
    def get(cls, name="Qwen/Qwen3-8B"):
        if name not in cls._instances:
            cls._instances[name] = cls(name)
        return cls._instances[name]

    @classmethod
    def unload(cls, name="Qwen/Qwen3-8B"):
        cls._instances.pop(name, None)
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def __init__(self, name="Qwen/Qwen3-8B"):
        self.tokenizer = AutoTokenizer.from_pretrained(name)
        self.model = AutoModelForCausalLM.from_pretrained(
            name,
            quantization_config=bnb_config,
            device_map="auto",
        )
        self.model.eval()

    def generate(self, messages, max_new_tokens=512, thinking=False):
        formatted = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=thinking,
        )
        inputs = self.tokenizer(formatted, return_tensors="pt").to(self.model.device)

        if thinking:
            sampling = dict(do_sample=True, temperature=0.6, top_p=0.95, top_k=20)
        else:
            sampling = dict(do_sample=False)

        with torch.inference_mode():
            output = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                pad_token_id=self.tokenizer.eos_token_id,
                **sampling,
            )

        new_tokens = output[0, inputs["input_ids"].shape[1]:]
        text = self.tokenizer.decode(new_tokens, skip_special_tokens=True)

        del inputs, output
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return strip_think(text)
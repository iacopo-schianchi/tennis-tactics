import gc
import re

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from google import genai

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)

THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)

GOOGLE_API_DEFAULT_MODEL = "gemini-3.5-flash-lite"
LOCAL_DEFAULT_MODEL = "Qwen/Qwen3-8B"

def strip_think(text):
    text = THINK_RE.sub("", text)
    if "<think>" in text:
        text = text.split("<think>")[0]
    return text.strip()


class Model:
    _instances = {}

    @classmethod
    def get(cls, name=None, local=True):
        name = name or (
            LOCAL_DEFAULT_MODEL if local else GOOGLE_API_DEFAULT_MODEL
        )
        key = (local, name)

        if key not in cls._instances:
            cls._instances[key] = cls(name, local=local)

        return cls._instances[key]

    @classmethod
    def unload(cls, name=None, local=True):
        name = name or (
            LOCAL_DEFAULT_MODEL if local else GOOGLE_API_DEFAULT_MODEL
        )
        cls._instances.pop((local, name), None)
        gc.collect()

        if local and torch.cuda.is_available():
            torch.cuda.empty_cache()

    def __init__(self, name=None, local=True):
        self.local = local
        self.name = name or (
            LOCAL_DEFAULT_MODEL if local else GOOGLE_API_DEFAULT_MODEL
        )

        if not local:
            self.client = genai.Client()
            return

        self.tokenizer = AutoTokenizer.from_pretrained(self.name)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.name,
            quantization_config=bnb_config,
            device_map="auto",
        )
        self.model.eval()

    def generate(self, messages, max_new_tokens=512, thinking=False):
        if not self.local:
            contents = []
            system_instruction = []

            for message in messages:
                role = message["role"]
                content = message["content"]

                if role == "system": system_instruction.append(content)
                else:
                    contents.append({
                        "role": "model" if role == "assistant" else "user",
                        "parts": [{"text": content}],
                    })

            config = {
                "max_output_tokens": max_new_tokens,
            }

            if system_instruction:
                config["system_instruction"] = "\n\n".join(system_instruction)

            response = self.client.models.generate_content(
                model=self.name,
                contents=contents,
                config=config,
            )

            return strip_think(response.text or "")

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
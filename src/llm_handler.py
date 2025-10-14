import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from src.config import LLM_MODEL, LLM_DEVICE, LLM_LOAD_IN_4BIT, MODEL_CACHE_DIR

class LLMHandler:
    def __init__(self, model_name=LLM_MODEL, device=LLM_DEVICE, load_in_4bit=LLM_LOAD_IN_4BIT):
        print(f"[INFO] Loading LLM model: {model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=MODEL_CACHE_DIR)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            cache_dir=MODEL_CACHE_DIR,
            device_map="auto" if device == "cuda" else None,
            load_in_4bit=load_in_4bit
        )
        self.device = device
        print("[INFO] Model successfully loaded")

    def generate(self, prompt, max_new_tokens=200, temperature=0.7):
        """
        Generates text from a given prompt.
        """
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id
            )
        text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return text

# Quick test
if __name__ == "__main__":
    handler = LLMHandler()
    prompt = "Write a short summary about Python and its uses."
    response = handler.generate(prompt)
    print("\n--- LLM Response ---\n")
    print(response)

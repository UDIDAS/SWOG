#!/usr/bin/env python3
"""
Llama 3.2 3B backend for the OAKG query app — load once, generate chat completions.

Kept free of Streamlit so it can be unit-tested and reused. The app wraps `load_llama()` in
st.cache_resource so the 3B model loads a single time and stays resident on the GPU.
"""
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# official gated repo first (token-gated), ungated mirror as fallback
REPOS = ["meta-llama/Llama-3.2-3B-Instruct", "unsloth/Llama-3.2-3B-Instruct"]


def load_llama():
    """Return (tokenizer, model, repo_name). Uses GPU bf16 when available."""
    last = None
    for repo in REPOS:
        try:
            tok = AutoTokenizer.from_pretrained(repo)
            try:                                            # transformers 5.x uses `dtype`
                model = AutoModelForCausalLM.from_pretrained(repo, dtype=torch.bfloat16)
            except TypeError:                               # older uses `torch_dtype`
                model = AutoModelForCausalLM.from_pretrained(repo, torch_dtype=torch.bfloat16)
            model = model.to("cuda:0" if torch.cuda.is_available() else "cpu").eval()
            return tok, model, repo
        except Exception as e:                              # try the next repo
            last = e
    raise last


def generate(tok, model, messages, max_new_tokens=320, temperature=0.3):
    """messages = [{'role': 'system'|'user'|'assistant', 'content': str}] -> assistant text."""
    inp = tok.apply_chat_template(messages, add_generation_prompt=True,
                                  return_tensors="pt", return_dict=True).to(model.device)
    with torch.no_grad():
        out = model.generate(**inp, max_new_tokens=max_new_tokens,
                             do_sample=temperature > 0, temperature=max(temperature, 1e-4),
                             top_p=0.9, pad_token_id=tok.eos_token_id)
    return tok.decode(out[0, inp["input_ids"].shape[1]:], skip_special_tokens=True).strip()

from collections import OrderedDict
from typing import List
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig


class LoRAM:
    def __init__(self, base_model_name: str, adapters_dict: dict[str,LoraConfig], max_loaded=2, device="cuda"):
        self.tokenizer = AutoTokenizer.from_pretrained(base_model_name)
        self.bases: List[AutoModelForCausalLM] = []
        for _ in range(max_loaded):
            m = AutoModelForCausalLM.from_pretrained(base_model_name).to(device)
            for name, cfg in adapters_dict.items():
                m.add_adapter(cfg, adapter_name=name)
            self.bases.append(m)


        self.max_loaded = max_loaded
        # Stores adapter names
        self.adapters = list(adapters_dict.keys())


        # adapter_idx: model_idx
        self._cache = OrderedDict()

    def swap_adapters(self, base_idx: int, adapter_idx: int):
        # Change to adapter_idx
        self.bases[base_idx].set_adapter(self.adapters[adapter_idx])
        self._cache[adapter_idx] = base_idx

    def switch_to(self, adapter_idx: int):
        # If already in cache, awesome:)
        if adapter_idx in self._cache:
            self._cache.move_to_end(adapter_idx)
            model_idx = self._cache[adapter_idx]
        else:
            # Check if over cap
            model_idx = self._cache.popitem(last=False)[1] if len(self._cache) == self.max_loaded else len(self._cache)
            self.swap_adapters(model_idx, adapter_idx)
        return self.bases[model_idx]

import os
from transformers import AutoTokenizer

class Tokenizer:
    """Wrapper around HuggingFace AutoTokenizer for prompt encoding and generation decoding."""

    def __init__(self, model_dir_or_id: str):
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_dir_or_id, local_files_only=True)
        except Exception:
            self.tokenizer = AutoTokenizer.from_pretrained(model_dir_or_id)
            
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

    @property
    def eos_token_id(self) -> int:
        return self.tokenizer.eos_token_id

    @property
    def bos_token_id(self) -> int:
        return self.tokenizer.bos_token_id

    def encode(self, text: str, add_special_tokens: bool = True) -> list[int]:
        return self.tokenizer.encode(text, add_special_tokens=add_special_tokens)

    def decode(self, token_ids: list[int], skip_special_tokens: bool = True) -> str:
        return self.tokenizer.decode(token_ids, skip_special_tokens=skip_special_tokens)

    def apply_chat_template(self, prompt, system_prompt: str = None, think: bool = False) -> str:
        """Formats prompt or message list with the model's chat template if available, else returns raw prompt."""
        if isinstance(prompt, str):
            if prompt.startswith("<|im_start|>") or "<|begin_of_text|>" in prompt:
                return prompt

            base_sys = system_prompt or "You are a helpful assistant."
            if think:
                sys_msg = (
                    f"{base_sys} "
                    "First think step by step inside <think>...</think> tags to plan your response, then output </think> and write your complete, detailed final answer in full."
                )
                messages = [
                    {"role": "system", "content": sys_msg},
                    {"role": "user", "content": prompt}
                ]
            elif system_prompt:
                messages = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt}
                ]
            else:
                messages = [{"role": "user", "content": prompt}]
        elif isinstance(prompt, list):
            messages = list(prompt)
            has_sys = any(m.get("role") == "system" for m in messages)
            if not has_sys:
                base_sys = system_prompt or "You are a helpful assistant."
                if think:
                    sys_msg = (
                        f"{base_sys} "
                        "First think step by step inside <think>...</think> tags to plan your response, then output </think> and write your complete, detailed final answer in full."
                    )
                    messages.insert(0, {"role": "system", "content": sys_msg})
                elif system_prompt:
                    messages.insert(0, {"role": "system", "content": system_prompt})
        else:
            return str(prompt)

        if hasattr(self.tokenizer, "apply_chat_template") and self.tokenizer.chat_template:
            try:
                formatted = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                if think:
                    formatted += "<think>\n"
                return formatted
            except Exception:
                pass
        
        # Fallback if chat template is not available
        if isinstance(prompt, list):
            return "\n".join(f"{m.get('role', 'user')}: {m.get('content', '')}" for m in messages)
        return prompt


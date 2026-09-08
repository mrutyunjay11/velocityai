from typing import Optional, List, Dict
from velocityai.generate import generate
from velocityai.tokenizer import Tokenizer
from velocityai.nn.transformer import LlamaModel

class Agent:
    """
    Autonomous AI Agent with a distinct persona, system prompt, and decoding hyperparameters.
    """
    def __init__(
        self,
        name: str,
        role: str,
        system_prompt: str,
        temperature: float = 0.0,
        top_p: float = 1.0,
        repetition_penalty: float = 1.0,
        code_mode: bool = False,
        max_tokens: int = 1024,
        color_code: str = "\033[36m"
    ):
        self.name = name
        self.role = role
        self.system_prompt = system_prompt
        self.temperature = temperature
        self.top_p = top_p
        self.repetition_penalty = repetition_penalty
        self.code_mode = code_mode
        self.max_tokens = max_tokens
        self.color_code = color_code
        self.reset_code = "\033[0m"

    def banner(self) -> str:
        return f"{self.color_code}\033[1m┌─ [{self.name}] {self.role}\033[0m"

    def footer(self) -> str:
        return f"{self.color_code}\033[1m└─ Done [{self.name}]\033[0m\n"

    def step(
        self,
        model: LlamaModel,
        tokenizer: Tokenizer,
        task: str,
        context: Optional[str] = None,
        callback=None,
        device: str = "cpu",
        sparse: bool = False,
        sparse_threshold: float = -3.5,
        stream: bool = True,
        auto_continue: bool = True,
        think: bool = False
    ) -> str:
        """Executes a reasoning/generation step for this agent."""
        prompt_text = task
        if context:
            prompt_text = f"Context and Prior Findings:\n{context}\n\nTask for {self.name}:\n{task}"

        token_cb = callback
        if stream and token_cb is None:
            token_cb = lambda tok: print(tok, end="", flush=True)

        res = generate(
            model=model,
            tokenizer=tokenizer,
            prompt=prompt_text,
            max_new_tokens=self.max_tokens,
            temperature=self.temperature,
            top_p=self.top_p,
            repetition_penalty=self.repetition_penalty,
            callback=token_cb,
            device=device,
            verbose=False,
            system_prompt=self.system_prompt,
            think=think,
            sparse=sparse,
            sparse_threshold=sparse_threshold,
            code_mode=self.code_mode,
            auto_continue=auto_continue
        )
        return res.get("text", "")


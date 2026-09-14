import time
import re
import numpy as np
import velocityai as vai
from velocityai.nn.transformer import LlamaModel
from velocityai.nn.kv_cache import KVCache
from velocityai.tokenizer import Tokenizer

def _detect_repetition_loop(tokens: list[int], min_len: int = 4, max_len: int = 256, min_repeats: int = 2) -> int:
    """Detects if recent generated tokens form an infinite repeating cycle (up to 256 tokens)."""
    n = len(tokens)
    max_search = min(max_len, n // min_repeats)
    for l in range(min_len, max_search + 1):
        target = tokens[-l:]
        repeats = 1
        for r in range(2, min_repeats + 1):
            if tokens[-r * l : -(r - 1) * l] == target:
                repeats += 1
            else:
                break
        if repeats >= min_repeats:
            return l
    return 0

def _detect_line_repetition(text: str, min_line_len: int = 12) -> bool:
    """
    Detects degenerative repetition loops:
    1. Consecutive duplicate lines (e.g. 'Step 16: The astronaut...' followed by 'Step 17: The astronaut...').
    2. Alternating 2-line oscillations (A B A B).
    3. Dense localized repetitions (same line >= 3 times in last 10 lines).
    4. Pathological global repetitions (same 20+ char line >= 4 times overall).
    Does NOT falsely trigger on valid code where two distinct functions share a line (e.g. partition calls).
    """
    raw_lines = [l.strip() for l in text.split("\n") if l.strip()]
    if len(raw_lines) < 2:
        return False

    cleaned_lines = []
    for raw in raw_lines:
        c = re.sub(
            r"^(?:[\s\*\-\•]*(?:step|item|point|part|phase)\s*\d+[:\.\)]?|[\s\*\-\•]*\d+[\.\:\)]|\*+|\-+|\•+)\s*(\*\*)?",
            "",
            raw,
            flags=re.IGNORECASE
        ).strip(" *-_").rstrip(".,:;!? ").lower()
        if len(c) >= min_line_len:
            cleaned_lines.append(c)

    if len(cleaned_lines) < 2:
        return False

    # Check 1: Consecutive duplicate line
    if cleaned_lines[-1] == cleaned_lines[-2]:
        return True

    # Check 2: Alternating 2-line oscillation (A B A B)
    if len(cleaned_lines) >= 4 and cleaned_lines[-1] == cleaned_lines[-3] and cleaned_lines[-2] == cleaned_lines[-4]:
        return True

    # Check 3: Dense localized repetition in recent window (>= 3 times in last 10 lines)
    recent = cleaned_lines[-10:]
    recent_counts = {}
    for l in recent:
        recent_counts[l] = recent_counts.get(l, 0) + 1
        if recent_counts[l] >= 3:
            return True

    # Check 4: Pathological global repetition (same 20+ char line appears >= 4 times)
    global_counts = {}
    for l in cleaned_lines:
        if len(l) >= 20:
            global_counts[l] = global_counts.get(l, 0) + 1
            if global_counts[l] >= 4:
                return True

    return False


def _is_output_incomplete(text: str, next_token: int, stop_token_ids: set) -> bool:
    """Checks if generated output was prematurely truncated before completing."""
    if next_token in stop_token_ids:
        return False
    
    # Check 0: Unclosed thinking block
    if "<think>" in text and "</think>" not in text:
        if any(marker in text for marker in ("Final Answer:", "\nAnswer:", "**Final Answer:**", "```")):
            return False
        return True

    # Check 0b: Thinking finished, but the answer portion after </think> is empty or barely started
    if "</think>" in text:
        answer_part = text.split("</think>", 1)[1].strip()
        if len(answer_part) < 15:
            return True
        text = answer_part

    # Check 1: Unclosed code blocks (odd number of triple backticks)
    if text.count("```") % 2 == 1:
        return True
        
    stripped = text.strip()
    if not stripped:
        return True
        
    # Check 2: Ends with dangling operators or incomplete syntax
    dangling_chars = (",", "(", "[", "{", "+", "-", "*", "/", "\\", ":", "=", ">", "<")
    if stripped.endswith(dangling_chars):
        return True
        
    # Check 3: Incomplete line-ending keywords
    words = stripped.split()
    if words and words[-1] in {"def", "class", "import", "from", "return", "elif", "else:", "if", "for", "while", "try:", "except:", "with", "and", "or", "the", "a", "an", "is", "of", "to", "in"}:
        return True
        
    # Check 4: Doesn't end with terminal punctuation
    if not stripped.endswith((".", "!", "?", "```", "\n", "}", "]", ")", '"', "'")):
        return True
        
    return False

def _sample_token(
    logits: np.ndarray,
    temperature: float = 0.0,
    top_p: float = 1.0,
    repetition_penalty: float = 1.0,
    past_tokens: list[int] = None
) -> int:
    logits = np.array(logits, dtype=np.float32, copy=True)
    if repetition_penalty != 1.0 and past_tokens:
        recent_tokens = set(past_tokens[-512:] if len(past_tokens) > 512 else past_tokens)
        for tid in recent_tokens:
            if tid < len(logits):
                if logits[tid] < 0:
                    logits[tid] *= repetition_penalty
                else:
                    logits[tid] /= repetition_penalty

    if temperature <= 1e-6:
        return int(np.argmax(logits))

    scaled_logits = (logits / temperature).astype(np.float64)
    scaled_logits -= np.max(scaled_logits)
    probs = np.exp(scaled_logits)
    sum_probs = np.sum(probs)
    if sum_probs > 0:
        probs /= sum_probs
    else:
        return int(np.argmax(logits))

    if top_p < 1.0:
        sorted_indices = np.argsort(probs)[::-1]
        sorted_probs = probs[sorted_indices]
        cumulative_probs = np.cumsum(sorted_probs)
        indices_to_remove = sorted_indices[cumulative_probs > top_p]
        if len(indices_to_remove) > 0 and len(indices_to_remove) < len(probs):
            probs[indices_to_remove] = 0.0
            sum_p = np.sum(probs)
            if sum_p > 0:
                probs /= sum_p
            else:
                return int(sorted_indices[0])

    return int(np.random.choice(len(probs), p=probs))

def generate(
    model: LlamaModel,
    tokenizer: Tokenizer,
    prompt,
    max_new_tokens: int = 512,
    temperature: float = 0.0,
    top_p: float = 1.0,
    repetition_penalty: float = 1.0,
    callback=None,
    device: str = "cpu",
    verbose: bool = False,
    system_prompt: str = None,
    think: bool = True,
    sparse: bool = False,
    sparse_threshold: float = -3.5,
    moe: bool = False,
    code_mode: bool = False,
    auto_continue: bool = True,
    max_continuations: int = 3
) -> dict:
    """
    Autoregressive text generation using VelocityAI LlamaModel with KV Cache.
    Supports thinking display by default, code_mode preset, dynamic sparsity, and dynamic context auto-continuation.
    """
    if code_mode and not system_prompt:
        system_prompt = (
            "You are an expert software developer. Write clean, correct, bug-free, and production-ready code. "
            "Do not hallucinate non-existent libraries or functions. Ensure all GUI elements and variables are properly initialized and imported. "
            "Keep any planning inside <think>...</think> brief, and always output </think> before writing code."
        )

    formatted_prompt = tokenizer.apply_chat_template(prompt, system_prompt=system_prompt, think=think)
    input_ids = tokenizer.encode(formatted_prompt)
    
    current_tokens = list(input_ids)
    generated_tokens = []
    
    stop_token_ids = {tokenizer.eos_token_id, 0}
    if hasattr(tokenizer, "tokenizer") and hasattr(tokenizer.tokenizer, "all_special_ids"):
        stop_token_ids.update(tokenizer.tokenizer.all_special_ids)
    
    # Initialize KV Cache with enough room for initial budget + continuations
    max_continuations_allowed = max_continuations if auto_continue else 0
    total_allowed_tokens = max_new_tokens * (max_continuations_allowed + 1)
    kv_cache = KVCache(
        num_layers=model.num_layers,
        num_kv_heads=model.num_kv_heads,
        max_seq_len=len(input_ids) + total_allowed_tokens + 128,
        head_dim=model.head_dim,
        device=device
    )
    
    # Fast native C++ decoder integration (cached on model)
    use_fast_decoder = (
        device == "cpu"
        and hasattr(model, 'layers')
        and hasattr(model.layers[0].self_attn, 'qkv_proj')
        and model.layers[0].self_attn.qkv_proj is not None
        and hasattr(kv_cache, 'k_tensors')
    )
    
    fast_decoder = None
    if use_fast_decoder:
        try:
            from _velocityai_c import FastLlamaDecoder
            lm_head_c = model.embed_tokens.weight._c if (model.tie_word_embeddings or model.lm_head is None) else model.lm_head.weight._c
            fast_decoder = FastLlamaDecoder(
                model.vocab_size,
                model.dim,
                model.num_layers,
                model.num_heads,
                model.num_kv_heads,
                model.head_dim,
                model.intermediate_size,
                kv_cache.max_seq_len,
                model.rms_norm_eps,
                model.rope_theta,
                model.embed_tokens.weight._c,
                model.norm.weight._c,
                lm_head_c
            )
            for i, layer in enumerate(model.layers):
                fast_decoder.add_layer(
                    layer.input_layernorm.weight._c,
                    layer.self_attn.qkv_proj.weight._c,
                    layer.self_attn.o_proj.weight._c,
                    layer.post_attention_layernorm.weight._c,
                    layer.mlp.gate_up_proj.weight._c,
                    layer.mlp.down_proj.weight._c,
                    kv_cache.k_tensors[i]._c,
                    kv_cache.v_tensors[i]._c,
                    layer.self_attn.qkv_proj.bias._c if layer.self_attn.qkv_proj.bias is not None else None
                )
            fast_decoder.set_sparse_mode(sparse, sparse_threshold)
        except Exception:
            fast_decoder = None

    # Prefill Phase
    gen_start_time = time.time()
    needs_sampling = (temperature > 1e-6 or repetition_penalty != 1.0 or top_p < 1.0)
    
    if fast_decoder is not None:
        if needs_sampling:
            logits_tensor = fast_decoder.prefill_logits(current_tokens)
            logits_np = logits_tensor.numpy()
            last_logits = logits_np[0, :] if logits_np.ndim == 2 else logits_np[:]
            next_token = _sample_token(last_logits, temperature, top_p, repetition_penalty, current_tokens)
        else:
            next_token = fast_decoder.prefill(current_tokens)
    else:
        # Python model fallback
        tokens_tensor = vai.tensor(np.array([current_tokens], dtype=np.int64)).to(device)
        logits = model(tokens_tensor, start_pos=0, last_token_only=True, kv_cache=kv_cache)
        logits_np = logits.numpy()
        last_logits = logits_np[0, -1, :] if logits_np.ndim == 3 else logits_np[-1, :]
        next_token = _sample_token(last_logits, temperature, top_p, repetition_penalty, current_tokens)
        
    ttft = time.time() - gen_start_time
    generated_tokens.append(next_token)
    current_tokens.append(next_token)
    
    if verbose:
        token_str = tokenizer.decode([next_token])
        print(f"[1/{max_new_tokens}] Generated token: {repr(token_str)}", flush=True)
        if callback:
            callback(token_str)
    elif callback:
        token_str = tokenizer.decode([next_token])
        callback(token_str)
        
    start_pos = len(input_ids)

    # Decode Phase with Dynamic Continuation
    continuations_performed = 0
    tokens_budget = max_new_tokens
    think_auto_closed = False

    while True:
        loop_stopped_by_cycle = False
        for _ in range(tokens_budget - 1 if continuations_performed == 0 else tokens_budget):
            if next_token in stop_token_ids:
                break

            # Check for degenerative repetition loops (both token cycles and numbered line repeats)
            if len(generated_tokens) >= 15:
                cycle_len = _detect_repetition_loop(generated_tokens)
                line_repeated = False
                if cycle_len == 0 and len(generated_tokens) % 8 == 0:
                    check_text = tokenizer.decode(generated_tokens)
                    if _detect_line_repetition(check_text, min_line_len=12):
                        line_repeated = True

                if cycle_len > 0 or line_repeated:
                    check_text = tokenizer.decode(generated_tokens)
                    # Case A: Loop occurred inside an unclosed <think> block
                    if "<think>" in check_text and "</think>" not in check_text and not think_auto_closed:
                        think_auto_closed = True
                        if cycle_len > 0:
                            while len(generated_tokens) >= 2 * cycle_len and generated_tokens[-cycle_len:] == generated_tokens[-2 * cycle_len : -cycle_len]:
                                generated_tokens = generated_tokens[:-cycle_len]
                        
                        # Close thinking cleanly and force the model to proceed to the answer
                        close_think_tokens = tokenizer.encode("\n</think>\n\n", add_special_tokens=False)
                        for ct in close_think_tokens:
                            generated_tokens.append(ct)
                            current_tokens.append(ct)
                            if callback:
                                callback(tokenizer.decode([ct]))
                            if fast_decoder is not None:
                                if not needs_sampling:
                                    next_token = fast_decoder.decode_step(ct, start_pos)
                                else:
                                    logits_c = fast_decoder.decode_step_logits(ct, start_pos)
                                    last_logits = vai.Tensor._wrap(logits_c).numpy()[0]
                                    next_token = _sample_token(last_logits, temperature, top_p, repetition_penalty, current_tokens)
                            else:
                                tokens_tensor = vai.tensor(np.array([[ct]], dtype=np.int64)).to(device)
                                logits = model(tokens_tensor, start_pos=start_pos, last_token_only=True, kv_cache=kv_cache)
                                logits_np = logits.numpy()
                                last_logits = logits_np[0, -1, :] if logits_np.ndim == 3 else logits_np[-1, :]
                                next_token = _sample_token(last_logits, temperature, top_p, repetition_penalty, current_tokens)
                            start_pos += 1
                        continue

                    # Case B: Loop occurred in final response
                    loop_stopped_by_cycle = True
                    if cycle_len > 0:
                        while len(generated_tokens) >= 2 * cycle_len and generated_tokens[-cycle_len:] == generated_tokens[-2 * cycle_len : -cycle_len]:
                            generated_tokens = generated_tokens[:-cycle_len]
                    break
                
            if fast_decoder is not None and not needs_sampling:
                next_token = fast_decoder.decode_step(next_token, start_pos)
            elif fast_decoder is not None:
                logits_c = fast_decoder.decode_step_logits(next_token, start_pos)
                last_logits = vai.Tensor._wrap(logits_c).numpy()[0]
                next_token = _sample_token(last_logits, temperature, top_p, repetition_penalty, current_tokens)
            else:
                tokens_tensor = vai.tensor(np.array([[next_token]], dtype=np.int64)).to(device)
                logits = model(tokens_tensor, start_pos=start_pos, last_token_only=True, kv_cache=kv_cache)
                logits_np = logits.numpy()
                last_logits = logits_np[0, -1, :] if logits_np.ndim == 3 else logits_np[-1, :]
                next_token = _sample_token(last_logits, temperature, top_p, repetition_penalty, current_tokens)
                
            generated_tokens.append(next_token)
            current_tokens.append(next_token)
            start_pos += 1
            
            if verbose:
                token_str = tokenizer.decode([next_token])
                print(f"[{len(generated_tokens)}/{total_allowed_tokens}] Generated token: {repr(token_str)}", flush=True)
                if callback:
                    callback(token_str)
            elif callback:
                token_str = tokenizer.decode([next_token])
                callback(token_str)

        # Check if generation terminated naturally or stopped by a loop
        if next_token in stop_token_ids or loop_stopped_by_cycle:
            break

        # Check for dynamic auto-continuation
        current_text = tokenizer.decode(generated_tokens)
        if auto_continue and not loop_stopped_by_cycle and continuations_performed < max_continuations and _is_output_incomplete(current_text, next_token, stop_token_ids):
            continuations_performed += 1
            tokens_budget = max_new_tokens
            if verbose:
                print(f"\n[🔄 Auto-continuing: incomplete response at {len(generated_tokens)} tokens...]", flush=True)
        else:
            break
    
    total_gen_time = time.time() - gen_start_time
    num_tokens = len(generated_tokens)
    tps = (num_tokens / total_gen_time) if total_gen_time > 0 else 0.0
    decode_time = total_gen_time - (ttft if ttft else 0.0)
    decode_tps = (num_tokens - 1) / decode_time if (decode_time > 0 and num_tokens > 1) else tps
    output_text = tokenizer.decode(generated_tokens)
    if "<think>" in output_text and "</think>" not in output_text:
        for marker in ("\nFinal Answer:", "\n\nFinal Answer:", "\nAnswer:", "\n\nAnswer:", "\n**Final Answer:**", "\n```", "\n\n```"):
            if marker in output_text:
                output_text = output_text.replace(marker, f"\n</think>\n{marker}", 1)
                break
    
    return {
        "text": output_text,
        "prompt": prompt,
        "input_tokens_count": len(input_ids),
        "output_tokens_count": num_tokens,
        "ttft": ttft,
        "total_time": total_gen_time,
        "decode_time": decode_time,
        "tps": tps,
        "decode_tps": decode_tps
    }

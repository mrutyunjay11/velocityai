#!/usr/bin/env python3
import sys
import time
import argparse
import select

class ThinkingStreamer:
    """
    Real-time streaming parser that separates <think>...</think> reasoning blocks
    from the final answer with elegant terminal UI styling.
    """
    def __init__(self, prefilled_think: bool = False):
        self.in_thinking = prefilled_think
        self.thinking_started = prefilled_think
        self.thinking_finished = False
        self.strip_leading_newlines = False
        self.pending_answer_header = False
        self.buffer = ""
        self.t_think_start = time.time() if prefilled_think else None
        self.think_duration = 0.0
        if prefilled_think:
            print("\033[36m\033[2m┌─ 💭 Thinking...\n│ ", end="", flush=True)

    def on_token(self, tok: str):
        self.buffer += tok

        # 1. Detect start of thinking if not prefilled
        if not self.thinking_started:
            if "<think>" in self.buffer:
                idx = self.buffer.find("<think>")
                prefix = self.buffer[:idx]
                if prefix.strip():
                    print(prefix, end="", flush=True)
                self.buffer = self.buffer[idx + len("<think>"):].lstrip("\n")
                self.in_thinking = True
                self.thinking_started = True
                self.t_think_start = time.time()
                print("\033[36m\033[2m┌─ 💭 Thinking...\n│ ", end="", flush=True)

        # 2. Inside thinking block
        if self.in_thinking:
            # If the model immediately started writing code with ``` right from the start
            stripped_buf = self.buffer.lstrip()
            if stripped_buf.startswith("```"):
                self.in_thinking = False
                self.thinking_finished = True
                print(f"Direct output (reasoning omitted)\n└─ Done\033[0m\n", flush=True)
                print(f"\n\033[1m💡 Answer:\033[0m\n{stripped_buf}", end="", flush=True)
                self.buffer = ""
                return

            close_idx = -1
            delimiter_len = 0
            is_implicit_marker = False
            
            if "</think>" in self.buffer:
                close_idx = self.buffer.find("</think>")
                delimiter_len = len("</think>")
            else:
                for marker in ("\nFinal Answer:", "\n\nFinal Answer:", "\nAnswer:", "\n\nAnswer:", "\n**Final Answer:**", "\n```", "\n\n```"):
                    if marker in self.buffer:
                        close_idx = self.buffer.find(marker)
                        delimiter_len = len(marker)
                        is_implicit_marker = True
                        break

            if close_idx != -1:
                think_text = self.buffer[:close_idx].rstrip("\n")
                formatted_think = think_text.replace("\n", "\n│ ")
                print(formatted_think, end="", flush=True)
                
                self.think_duration = time.time() - (self.t_think_start or time.time())
                print(f"\n└─ Done thinking ({self.think_duration:.2f}s)\033[0m\n", end="", flush=True)
                
                self.in_thinking = False
                self.thinking_finished = True
                self.strip_leading_newlines = True
                self.pending_answer_header = True
                
                if is_implicit_marker:
                    marker_text = self.buffer[close_idx:close_idx + delimiter_len].strip()
                    remainder = self.buffer[close_idx + delimiter_len:].lstrip("\n")
                    after_text = f"{marker_text} {remainder}".strip()
                else:
                    after_text = self.buffer[close_idx + delimiter_len:].lstrip("\n")

                if after_text:
                    self.strip_leading_newlines = False
                    self.pending_answer_header = False
                    print(f"\n\033[1m💡 Answer:\033[0m\n{after_text}", end="", flush=True)
                self.buffer = ""
            else:
                # Stream out safe portion of buffer (keep last 18 chars in case a marker is split across tokens)
                if len(self.buffer) > 18:
                    safe = self.buffer[:-18]
                    self.buffer = self.buffer[-18:]
                    formatted = safe.replace("\n", "\n│ ")
                    print(formatted, end="", flush=True)
            return

        # 3. Strip extra newlines immediately following the </think> tag
        if self.strip_leading_newlines:
            self.buffer = self.buffer.lstrip("\n")
            if self.buffer:
                self.strip_leading_newlines = False
                if self.pending_answer_header:
                    print(f"\n\033[1m💡 Answer:\033[0m\n", end="", flush=True)
                    self.pending_answer_header = False
                print(self.buffer, end="", flush=True)
                self.buffer = ""
            return

        # 4. Normal response streaming
        if self.thinking_finished or self.thinking_started:
            if self.buffer:
                if self.pending_answer_header:
                    print(f"\n\033[1m💡 Answer:\033[0m\n", end="", flush=True)
                    self.pending_answer_header = False
                print(self.buffer, end="", flush=True)
                self.buffer = ""
        else:
            if len(self.buffer) > 10:
                print(self.buffer, end="", flush=True)
                self.buffer = ""

    def flush(self):
        if self.in_thinking:
            formatted = self.buffer.rstrip("\n").replace("\n", "\n│ ")
            print(formatted, end="", flush=True)
            self.think_duration = time.time() - (self.t_think_start or time.time())
            print(f"\n└─ Done thinking ({self.think_duration:.2f}s)\033[0m\n", flush=True)
        elif self.buffer:
            if self.pending_answer_header:
                print(f"\n\033[1m💡 Answer:\033[0m\n", end="", flush=True)
                self.pending_answer_header = False
            out = self.buffer.lstrip("\n") if self.strip_leading_newlines else self.buffer
            print(out, end="", flush=True)
        self.buffer = ""


def main():
    parser = argparse.ArgumentParser(
        description="VelocityAI CLI: High-Performance Apple Silicon LLM Engine",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument(
        "prompt",
        nargs="?",
        default=None,
        help="Prompt to generate text for (e.g. 'Explain relativity in simple terms')"
    )
    parser.add_argument(
        "--model", "-m",
        default="HuggingFaceTB/SmolLM2-360M-Instruct",
        help="Model ID or directory path (default: HuggingFaceTB/SmolLM2-360M-Instruct)"
    )
    parser.add_argument(
        "--max-tokens", "-n",
        type=int,
        default=2048,
        help="Maximum tokens to generate (default: 2048, up to 8192 for SmolLM2)"
    )
    parser.add_argument(
        "--temperature", "-t",
        type=float,
        default=None,
        help="Sampling temperature: 0.0 for deterministic/greedy, >0 for stochastic (default: 0.2, or 0.15 with --code)"
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=None,
        help="Nucleus sampling top-p threshold (default: 0.95)"
    )
    parser.add_argument(
        "--repetition-penalty", "-rp",
        type=float,
        default=None,
        help="Repetition penalty factor: >1.0 discourages repeating tokens (default: 1.12, or 1.15 with --code)"
    )

    # Task Presets
    parser.add_argument(
        "--code",
        action="store_true",
        help="Enable specialized coding preset (auto-calibrates temp=0.15, rep_penalty=1.15, strict syntax)"
    )

    # Hardware Device Toggles
    parser.add_argument(
        "--device", "-d",
        choices=["cpu", "gpu", "metal"],
        default="cpu",
        help="Hardware compute target: 'cpu' (multi-core NEON) or 'gpu'/'metal' (Metal Performance Shaders)"
    )
    parser.add_argument(
        "--cpu",
        action="store_const",
        dest="device",
        const="cpu",
        help="Target Apple Silicon CPU (Multi-core NEON SIMD)"
    )
    parser.add_argument(
        "--gpu",
        action="store_const",
        dest="device",
        const="metal",
        help="Target Apple Silicon GPU (Metal unified memory)"
    )

    # Architecture & Compute Modes
    parser.add_argument(
        "--dense",
        action="store_true",
        help="Force full dense FP16 computation (100%% parameter activation, lossless)"
    )
    parser.add_argument(
        "--sparse", "-sp",
        action="store_true",
        help="Enable dynamic activation sparsity (skips cold SwiGLU neurons for higher speed)"
    )
    parser.add_argument(
        "--sparse-threshold",
        type=float,
        default=-3.5,
        help="Cutoff threshold for gate activation pruning (default: -3.5)"
    )
    parser.add_argument(
        "--moe",
        action="store_true",
        help="Enable Mixture-of-Experts (MoE) routing mode"
    )

    # Autonomous Multi-Agent & Web Tools
    parser.add_argument(
        "--team", "--agents",
        action="store_true",
        dest="team",
        help="Activate collaborative Multi-Agent Team (Architect + Developer + Reviewer + Creative + Web)"
    )
    parser.add_argument(
        "--web", "--browser", "--allow-internet",
        action="store_true",
        dest="allow_internet",
        help="Explicitly grant permission for agents to access the internet (search & fetch web documentation)"
    )

    parser.add_argument(
        "--chat", "-c",
        action="store_true",
        help="Start an interactive multi-turn terminal chat session"
    )
    parser.add_argument(
        "--think", "-th",
        action="store_true",
        dest="think",
        default=None,
        help="Display the model's live reasoning / thinking process in a styled terminal box (default: True for chat, False for --code)"
    )
    parser.add_argument(
        "--no-think",
        action="store_false",
        dest="think",
        help="Disable the live reasoning / thinking process display"
    )
    parser.add_argument(
        "--auto-continue",
        action="store_true",
        default=True,
        help="Automatically continue generation if output is incomplete or truncated (default: True)"
    )
    parser.add_argument(
        "--no-auto-continue",
        action="store_false",
        dest="auto_continue",
        help="Disable automatic continuation on truncated outputs"
    )
    parser.add_argument(
        "--no-stream",
        action="store_true",
        help="Disable live token streaming to terminal"
    )

    args = parser.parse_args()

    if not args.chat and not args.prompt:
        parser.print_help()
        sys.exit(1)


    # Preset calibration: apply --code defaults if not explicitly set by user
    is_code_mode = args.code
    args.think = (False if is_code_mode else True) if args.think is None else args.think
    active_temp = args.temperature if args.temperature is not None else (0.15 if is_code_mode else 0.2)
    active_top_p = args.top_p if args.top_p is not None else (0.95 if is_code_mode else 0.95)
    active_rep_penalty = args.repetition_penalty if args.repetition_penalty is not None else (1.15 if is_code_mode else 1.12)
    active_device = args.device
    active_sparse = args.sparse and not args.dense
    active_moe = args.moe
    is_team_mode = args.team
    allow_internet = args.allow_internet

    print("=" * 65)
    print("  🚀 VelocityAI Inference Engine (Apple Silicon M-Series Native)")
    print("=" * 65)

    import velocityai as vai
    from velocityai.tokenizer import Tokenizer
    from velocityai.generate import generate
    from velocityai.agents import AgentTeam

    # Multi-Agent Team Engine banner
    if is_team_mode:
        print("[*] Engine Mode:     🤝 MULTI-AGENT TEAM (Architect + Dev + Reviewer + Creative + Web)")
    else:
        print("[*] Engine Mode:     👤 SINGLE AGENT")

    # Internet Permission banner
    if allow_internet:
        print("[*] Internet Access: 🌐 GRANTED (Web Search & Doc Fetching Enabled)")
    else:
        print("[*] Internet Access: 🔒 OFFLINE / PRIVATE (Zero Network Calls)")

    # Hardware target banner
    if active_device in ("gpu", "metal"):
        print("[*] Hardware Target: 🍏 Apple Silicon GPU (Metal Unified Memory)")
    else:
        print("[*] Hardware Target: ⚡ Apple Silicon CPU (Multi-core NEON SIMD)")

    # Architecture banner
    if active_moe:
        print("[*] Architecture:    🔀 MIXTURE OF EXPERTS (Top-K Active Routing)")
    elif active_sparse:
        print(f"[*] Architecture:    ⚡ DYNAMIC SPARSITY (SwiGLU Cutoff: {args.sparse_threshold})")
    else:
        print("[*] Architecture:    🛡️  FULL DENSE (100% Parameters, Lossless FP16)")

    # Task preset banner
    if is_code_mode:
        print(f"[*] Task Preset:     💻 CODE OPTIMIZED (temp={active_temp}, rep_penalty={active_rep_penalty}, strict syntax)")
    else:
        print(f"[*] Task Preset:     💬 GENERAL CHAT (temp={active_temp}, rep_penalty={active_rep_penalty})")

    # Thinking mode banner
    if args.think:
        print("[*] Thinking Mode:    💭 ENABLED (Step-by-step reasoning)")
    else:
        print("[*] Thinking Mode:    ⚡ FAST (Direct response, reasoning omitted)")

    # Dynamic auto-continuation banner
    if args.auto_continue:
        print("[*] Auto-Continue:    🔄 ENABLED (Dynamic context completion)")
    else:
        print("[*] Auto-Continue:    ⏹️ DISABLED (Hard token cutoff)")

    print(f"[*] Loading model: {args.model}...")
    t0 = time.time()
    model, config = vai.load_huggingface_model(args.model)
    tokenizer = Tokenizer(args.model)
    team = AgentTeam(allow_internet=allow_internet)
    load_time = time.time() - t0
    print(f"[✓] Model ready in {load_time:.2f}s!\n")

    def run_one(prompt_or_messages):
        if is_team_mode:
            if isinstance(prompt_or_messages, list):
                user_text = next((m["content"] for m in reversed(prompt_or_messages) if m.get("role") == "user"), str(prompt_or_messages))
            else:
                user_text = str(prompt_or_messages)

            team.set_internet_permission(allow_internet)
            team_res = team.run(
                model=model,
                tokenizer=tokenizer,
                user_request=user_text,
                device=active_device,
                sparse=active_sparse,
                sparse_threshold=args.sparse_threshold,
                stream=not args.no_stream
            )
            return team_res["final"]

        if not args.chat:
            print(f"User: {prompt_or_messages}")

        streamer = None
        callback = None
        if not args.no_stream:
            if args.think:
                streamer = ThinkingStreamer(prefilled_think=True)
                callback = streamer.on_token
            else:
                print("VelocityAI: ", end="", flush=True)
                callback = lambda tok: print(tok, end="", flush=True)
        elif not args.think:
            print("VelocityAI: ", end="", flush=True)

        res = generate(
            model=model,
            tokenizer=tokenizer,
            prompt=prompt_or_messages,
            max_new_tokens=args.max_tokens,
            temperature=active_temp,
            top_p=active_top_p,
            repetition_penalty=active_rep_penalty,
            callback=callback,
            device=active_device,
            verbose=False,
            think=args.think,
            sparse=active_sparse,
            sparse_threshold=args.sparse_threshold,
            moe=active_moe,
            code_mode=is_code_mode,
            auto_continue=args.auto_continue
        )
        if streamer:
            streamer.flush()
            print()
        elif args.no_stream:
            if args.think:
                text = res["text"]
                if "</think>" in text:
                    think_part, answer_part = text.split("</think>", 1)
                    formatted_think = think_part.strip().replace("\n", "\n│ ")
                    print(f"\033[36m\033[2m┌─ 💭 Thinking...\n│ {formatted_think}\n└─ Done thinking\033[0m\n\n\033[1m💡 Answer:\033[0m\n{answer_part.strip()}\n")
                else:
                    print(text)
            else:
                print(res["text"])
        else:
            print()

        ttft = res.get("ttft", 0.0)
        decode_tps = res.get("decode_tps", res["tps"])
        total_tokens = res["output_tokens_count"]
        print("-" * 65)
        print(f"⚡ Stats: {total_tokens} tokens | TTFT: {ttft:.3f}s | Decode: {decode_tps:.2f} tok/s | Total: {res['tps']:.2f} tok/s")
        print("-" * 65 + "\n")
        return res.get("text", "")

    if args.chat:
        print("💬 Interactive Chat Mode started (Type 'exit', 'quit', or 'q' to stop)")
        print("💡 Live Slash Commands:")
        print("   /team    - Toggle Multi-Agent Team Mode (Architect + Dev + Reviewer)")
        print("   /web     - Toggle Internet Access on/off (/web on, /web off)")
        print("   /code    - Toggle Code Mode on/off (temp=0.15, rep_penalty=1.15, precision prompt)")
        print("   /sparse  - Toggle Dynamic Sparsity on/off")
        print("   /cpu     - Switch compute target to CPU (NEON)")
        print("   /gpu     - Switch compute target to GPU (Metal)")
        print("   /status  - Show active settings and history turns")
        print("   /clear   - Clear conversation history")
        print("   /help    - Show slash command list")
        if args.think:
            print("💭 Live Thinking Mode: ENABLED")
        print("-" * 65)
        conversation_history = []
        while True:
            try:
                user_input = input("You > ")
                # If multi-line text was pasted, drain remaining lines from stdin buffer
                if sys.stdin.isatty():
                    extra_lines = []
                    while select.select([sys.stdin], [], [], 0.05)[0]:
                        line = sys.stdin.readline()
                        if not line:
                            break
                        extra_lines.append(line.rstrip("\r\n"))
                    if extra_lines:
                        user_input = user_input + "\n" + "\n".join(extra_lines)
                user_input = user_input.strip()
                if not user_input:
                    continue

                cmd = user_input.lower().strip()
                if cmd in ("exit", "quit", "q"):
                    print("Exiting chat. Goodbye!")
                    break
                if cmd in ("/clear", "/reset", "clear", "reset"):
                    conversation_history.clear()
                    print("[✓] Conversation history cleared.\n")
                    continue
                if cmd == "/team":
                    is_team_mode = not is_team_mode
                    state = "🤝 ENABLED (Architect + Dev + Reviewer / Creative / Web)" if is_team_mode else "👤 DISABLED (Single Agent)"
                    print(f"[✓] Multi-Agent Team Mode: {state}\n")
                    continue
                if cmd in ("/web on", "web on"):
                    allow_internet = True
                    team.set_internet_permission(True)
                    print("[✓] Internet Access: 🌐 GRANTED (Web Search & Doc Fetching Enabled)\n")
                    continue
                if cmd in ("/web off", "web off"):
                    allow_internet = False
                    team.set_internet_permission(False)
                    print("[✓] Internet Access: 🔒 REVOKED (100% Offline & Private)\n")
                    continue
                if cmd == "/web":
                    allow_internet = not allow_internet
                    team.set_internet_permission(allow_internet)
                    state = "🌐 GRANTED" if allow_internet else "🔒 REVOKED (Offline)"
                    print(f"[✓] Internet Access: {state}\n")
                    continue
                if cmd == "/code":
                    is_code_mode = not is_code_mode
                    if is_code_mode:
                        active_temp = 0.15
                        active_rep_penalty = 1.15
                        active_top_p = 0.95
                        print("[✓] Code Mode: 💻 ENABLED (temp=0.15, rep_penalty=1.15, top_p=0.95, precision syntax)\n")
                    else:
                        active_temp = args.temperature if args.temperature is not None else 0.2
                        active_rep_penalty = args.repetition_penalty if args.repetition_penalty is not None else 1.12
                        active_top_p = args.top_p if args.top_p is not None else 0.95
                        print("[✓] Code Mode: 💬 DISABLED (Switched to standard conversational chat)\n")
                    continue
                if cmd == "/sparse":
                    active_sparse = not active_sparse
                    state = f"⚡ ENABLED (cutoff: {args.sparse_threshold})" if active_sparse else "🛡️ DISABLED (Full Dense FP16)"
                    print(f"[✓] Dynamic Sparsity: {state}\n")
                    continue
                if cmd in ("/gpu", "/metal"):
                    active_device = "metal"
                    print("[✓] Compute Target: 🍏 Switched to Apple Silicon GPU (Metal Unified Memory)\n")
                    continue
                if cmd == "/cpu":
                    active_device = "cpu"
                    print("[✓] Compute Target: ⚡ Switched to Apple Silicon CPU (Multi-core NEON)\n")
                    continue
                if cmd in ("/think on", "think on"):
                    args.think = True
                    print("[✓] Live Thinking Mode: 💭 ENABLED (Step-by-step reasoning)\n")
                    continue
                if cmd in ("/think off", "think off"):
                    args.think = False
                    print("[✓] Live Thinking Mode: ⚡ DISABLED (Fast direct output)\n")
                    continue
                if cmd == "/think":
                    args.think = not args.think
                    state = "💭 ENABLED" if args.think else "⚡ DISABLED"
                    print(f"[✓] Live Thinking Mode: {state}\n")
                    continue
                if cmd in ("/autocontinue", "/continue"):
                    args.auto_continue = not args.auto_continue
                    state = "🔄 ENABLED (Dynamic context completion)" if args.auto_continue else "⏹️ DISABLED (Hard cutoff)"
                    print(f"[✓] Dynamic Auto-Continuation: {state}\n")
                    continue
                if cmd in ("/status", "/info"):
                    print("-" * 48)
                    print(f"[*] Engine Mode:     {'🤝 MULTI-AGENT TEAM' if is_team_mode else '👤 SINGLE AGENT'}")
                    print(f"[*] Internet Access: {'🌐 GRANTED' if allow_internet else '🔒 OFFLINE / PRIVATE'}")
                    print(f"[*] Task Preset:     {'💻 CODE OPTIMIZED' if is_code_mode else '💬 GENERAL CHAT'}")
                    print(f"[*] Thinking Mode:   {'💭 ENABLED' if args.think else '⚡ DISABLED'}")
                    print(f"[*] Auto-Continue:   {'🔄 ENABLED' if args.auto_continue else '⏹️ DISABLED'}")
                    print(f"[*] Target Hardware: {'🍏 GPU (Metal)' if active_device in ('gpu', 'metal') else '⚡ CPU (NEON)'}")
                    print(f"[*] Architecture:    {'🔀 MoE' if active_moe else ('⚡ DYNAMIC SPARSITY' if active_sparse else '🛡️ FULL DENSE')}")
                    print(f"[*] Temperature:     {active_temp}")
                    print(f"[*] Rep Penalty:     {active_rep_penalty}")
                    print(f"[*] Top-P:           {active_top_p}")
                    print(f"[*] History Turns:   {len(conversation_history) // 2}")
                    print("-" * 48 + "\n")
                    continue
                if cmd in ("/help", "help"):
                    print("Available slash commands:")
                    print("  /team         - Toggle Multi-Agent Team Mode")
                    print("  /web          - Toggle Internet Access on/off (/web on, /web off)")
                    print("  /code         - Toggle Code Mode on/off")
                    print("  /think        - Toggle Thinking Mode on/off (/think on, /think off)")
                    print("  /autocontinue - Toggle Dynamic Context Auto-Continuation on/off")
                    print("  /sparse       - Toggle Dynamic Sparsity on/off")
                    print("  /cpu          - Switch compute to CPU (NEON)")
                    print("  /gpu          - Switch compute to GPU (Metal)")
                    print("  /status       - Show active settings")
                    print("  /clear        - Clear conversation history")
                    print("  /help         - Show this help message")
                    print("  exit          - Exit chat session\n")
                    continue

                conversation_history.append({"role": "user", "content": user_input})
                reply = run_one(conversation_history)
                if reply:
                    conversation_history.append({"role": "assistant", "content": reply})
            except (KeyboardInterrupt, EOFError):
                print("\nExiting chat.")
                break
    else:
        run_one(args.prompt)

if __name__ == "__main__":
    main()

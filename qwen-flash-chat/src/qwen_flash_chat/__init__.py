"""Chat with Qwen Flash locally from the terminal, using MLX."""

import argparse
import sys
from pathlib import Path

DEFAULT_MODEL_DIR = Path(__file__).resolve().parents[2] / "models" / "qwen-flash-mlx"
DEFAULT_SYSTEM_PROMPT = "You are a helpful assistant"

# Sampling settings from the Qwen model card.
THINKING_SAMPLING = {"temperature": 1.0, "top_p": 0.95, "top_k": 20, "min_p": 0.0}
NON_THINKING_SAMPLING = {
    "temperature": 0.7,
    "top_p": 0.8,
    "top_k": 20,
    "min_p": 0.0,
    "presence_penalty": 1.5,
}

DIM = "\033[2m"
BOLD = "\033[1m"
RESET = "\033[0m"

HELP_TEXT = """Commands:
  /think on|off        turn thinking on or off
  /effort low|medium|xhigh  set reasoning effort (Flash Next)
  /reset               clear the conversation
  /help                show this help
  /quit                exit (Ctrl-D works too)
Press Ctrl-C while the model is answering to stop it."""


def stream_reply(model, processor, prompt: str, thinking: bool, max_tokens: int) -> str:
    """Stream one reply to the terminal and return the answer without the thinking part."""
    from mlx_vlm import stream_generate

    sampling = THINKING_SAMPLING if thinking else NON_THINKING_SAMPLING
    if thinking and model.config.model_type == "qwen3_5_moe":
        sampling = {**sampling, "presence_penalty": 1.5}
    in_thinking = thinking
    answer = []
    pending = ""
    end_marker = "</think>"
    last_chunk = None

    if in_thinking:
        print(DIM, end="", flush=True)

    try:
        for chunk in stream_generate(model, processor, prompt, max_tokens=max_tokens, **sampling):
            last_chunk = chunk
            text = chunk.text
            if in_thinking:
                pending += text
                thought, marker, text = pending.partition(end_marker)
                if not marker:
                    # Keep any partial delimiter until the next chunk arrives.
                    boundary = pending.rfind("<")
                    if boundary < 0 or not end_marker.startswith(pending[boundary:]):
                        boundary = len(pending)
                    print(pending[:boundary], end="", flush=True)
                    pending = pending[boundary:]
                    continue
                print(f"{thought}{RESET}", end="", flush=True)
                in_thinking = False
            print(text, end="", flush=True)
            answer.append(text)
    except KeyboardInterrupt:
        print(f"{RESET}\n[stopped]", end="")

    print(RESET)
    if last_chunk is not None:
        print(
            f"{DIM}[prompt: {last_chunk.prompt_tokens} tokens at {last_chunk.prompt_tps:.0f} tok/s, "
            f"reply: {last_chunk.generation_tokens} tokens at {last_chunk.generation_tps:.1f} tok/s, "
            f"peak memory: {last_chunk.peak_memory:.1f} GB]{RESET}"
        )

    return "".join(answer).strip()


def parse_args():
    parser = argparse.ArgumentParser(description="Chat with Qwen Flash (MLX).")
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    parser.add_argument("--no-think", action="store_true", help="start with thinking turned off")
    parser.add_argument("--effort", choices=["low", "medium", "xhigh"])
    parser.add_argument("--max-tokens", type=int, default=16384)
    parser.add_argument("--system", default=DEFAULT_SYSTEM_PROMPT, help="system prompt")
    return parser.parse_args()


def main():
    args = parse_args()
    if not (args.model_dir / "config.json").is_file():
        sys.exit(f"Model pack not found at {args.model_dir}. See README.md to download it.")

    print(f"Loading model from {args.model_dir} ...", flush=True)
    from mlx_vlm import load

    model, processor = load(str(args.model_dir))

    thinking = not args.no_think
    supports_effort = model.config.model_type == "qwen4_exp"
    if args.effort and not supports_effort:
        sys.exit("This model does not support reasoning effort. Use --no-think to disable thinking.")
    effort = args.effort or "medium"
    system_message = {"role": "system", "content": args.system}
    messages = [system_message]

    print(f"Ready. Thinking: {'on' if thinking else 'off'}. Type /help for commands.\n")

    while True:
        try:
            user_input = input(f"{BOLD}you>{RESET} ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if not user_input:
            continue
        if user_input in ("/quit", "/exit"):
            break
        if user_input == "/help":
            print(HELP_TEXT)
            continue
        if user_input == "/reset":
            messages = [system_message]
            print("Conversation cleared.")
            continue
        if user_input.split()[0] == "/think":
            if user_input not in ("/think on", "/think off"):
                print("Use /think on or /think off.")
                continue
            thinking = user_input.split()[-1] != "off"
            print(f"Thinking {'on' if thinking else 'off'}.")
            continue
        if user_input.split()[0] == "/effort":
            if not supports_effort:
                print("This model supports /think on or /think off.")
                continue
            requested = user_input.split()[-1]
            if len(user_input.split()) == 2 and requested in ("low", "medium", "xhigh"):
                effort = requested
                print(f"Reasoning effort: {effort}.")
            else:
                print("Use /effort low, /effort medium, or /effort xhigh.")
            continue

        messages.append({"role": "user", "content": user_input})
        prompt = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
            enable_thinking=thinking, reasoning_effort=effort,
        )
        print(f"{BOLD}qwen>{RESET} ", end="", flush=True)
        answer = stream_reply(model, processor, prompt, thinking, args.max_tokens)
        if answer:
            messages.append({"role": "assistant", "content": answer})
        else:
            messages.pop()

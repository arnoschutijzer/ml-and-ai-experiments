"""Chat with Ternary Bonsai 2.5 27B locally from the terminal, using MLX."""

import argparse
import sys
from pathlib import Path

DEFAULT_MODEL_DIR = Path(__file__).resolve().parents[2] / "models" / "bonsai-25-mlx"
DEFAULT_SYSTEM_PROMPT = "You are a helpful assistant"

# Sampling settings recommended by PrismML for the 27B model.
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
  /effort xhigh|medium set reasoning effort
  /reset               clear the conversation
  /help                show this help
  /quit                exit (Ctrl-D works too)
Press Ctrl-C while the model is answering to stop it."""


def load(model_dir: Path):
    """Load the pack with its bundled loader, which applies the Hadamard transforms."""
    sys.path.insert(0, str(model_dir / "runtime"))
    from vision_artifact import load_vl_model

    model, processor, _ = load_vl_model(model_dir)
    return model, processor


def build_template(model_dir: Path):
    from jinja2.sandbox import ImmutableSandboxedEnvironment

    environment = ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True)
    return environment.from_string((model_dir / "chat_template.jinja").read_text())


def render_prompt(template, messages, thinking: bool, effort: str) -> str:
    return template.render(
        messages=messages,
        add_generation_prompt=True,
        enable_thinking=thinking,
        reasoning_effort=effort,
    )


def stream_reply(model, processor, prompt: str, thinking: bool, max_tokens: int) -> str:
    """Stream one reply to the terminal and return the answer without the thinking part."""
    from mlx_vlm import stream_generate

    sampling = THINKING_SAMPLING if thinking else NON_THINKING_SAMPLING
    in_thinking = thinking
    full_text = ""
    last_chunk = None

    if in_thinking:
        print(f"{DIM}", end="", flush=True)

    try:
        for chunk in stream_generate(model, processor, prompt, max_tokens=max_tokens, **sampling):
            last_chunk = chunk
            full_text += chunk.text
            if in_thinking and "</think>" in full_text:
                thought_tail, _, answer_start = chunk.text.partition("</think>")
                print(f"{thought_tail}{RESET}\n", end="", flush=True)
                print(answer_start.lstrip("\n"), end="", flush=True)
                in_thinking = False
            else:
                print(chunk.text, end="", flush=True)
    except KeyboardInterrupt:
        print(f"{RESET}\n[stopped]", end="")

    print(RESET)
    if last_chunk is not None:
        print(
            f"{DIM}[prompt: {last_chunk.prompt_tokens} tokens at {last_chunk.prompt_tps:.0f} tok/s, "
            f"reply: {last_chunk.generation_tokens} tokens at {last_chunk.generation_tps:.1f} tok/s, "
            f"peak memory: {last_chunk.peak_memory:.1f} GB]{RESET}"
        )

    answer = full_text.split("</think>", 1)[-1] if thinking else full_text
    return answer.strip()


def parse_args():
    parser = argparse.ArgumentParser(description="Chat with Ternary Bonsai 2.5 27B (MLX).")
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    parser.add_argument("--no-think", action="store_true", help="start with thinking turned off")
    parser.add_argument("--effort", choices=["xhigh", "medium"], default="medium")
    parser.add_argument("--max-tokens", type=int, default=16384)
    parser.add_argument("--system", default=DEFAULT_SYSTEM_PROMPT, help="system prompt")
    return parser.parse_args()


def main():
    args = parse_args()
    if not (args.model_dir / "config.json").is_file():
        sys.exit(f"Model pack not found at {args.model_dir}. See README.md to download it.")

    print(f"Loading model from {args.model_dir} ...", flush=True)
    model, processor = load(args.model_dir)
    template = build_template(args.model_dir)

    thinking = not args.no_think
    effort = args.effort
    system_message = {"role": "system", "content": args.system}
    messages = [system_message]

    print(f"Ready. Thinking: {'on' if thinking else 'off'}, effort: {effort}. Type /help for commands.\n")

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
        if user_input.startswith("/think"):
            thinking = user_input.split()[-1] != "off"
            print(f"Thinking {'on' if thinking else 'off'}.")
            continue
        if user_input.startswith("/effort"):
            requested = user_input.split()[-1]
            if requested in ("xhigh", "medium"):
                effort = requested
                print(f"Reasoning effort: {effort}.")
            else:
                print("Use /effort xhigh or /effort medium.")
            continue

        messages.append({"role": "user", "content": user_input})
        prompt = render_prompt(template, messages, thinking, effort)
        print(f"{BOLD}bonsai>{RESET} ", end="", flush=True)
        answer = stream_reply(model, processor, prompt, thinking, args.max_tokens)
        messages.append({"role": "assistant", "content": answer})

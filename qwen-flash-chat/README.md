# qwen-flash-chat

A local terminal chat for Apple Silicon. The default setup uses Qwen3.5-35B-A3B, the open-weight counterpart of Qwen3.5 Flash. Its 4-bit pack is about 20.4 GB. Generation also needs memory for the conversation. [Qwen model card](https://huggingface.co/Qwen/Qwen3.5-35B-A3B) · [MLX pack](https://huggingface.co/mlx-community/Qwen3.5-35B-A3B-4bit)

## Setup and chat

Run these commands from this directory.

```sh
uv run hf download mlx-community/Qwen3.5-35B-A3B-4bit --local-dir models/qwen-flash-mlx
uv run qwen-flash-chat
```

Type `/think on` or `/think off` to change thinking mode, `/reset` to clear the conversation, and `/quit` to exit. Ctrl-C stops the current reply. Use `--no-think`, `--system "…"`, or `--max-tokens N` at startup.

## Qwen3.8 Flash Next

On a Mac with room for the 112 GB pack and the conversation, use the following commands. This model also supports `--effort low|medium|xhigh` and the matching `/effort` command. [MLX pack](https://huggingface.co/mlx-community/Qwen3.8-Flash-Next-4bit)

```sh
uv run hf download mlx-community/Qwen3.8-Flash-Next-4bit --local-dir models/qwen38-flash
uv run qwen-flash-chat --model-dir models/qwen38-flash
```

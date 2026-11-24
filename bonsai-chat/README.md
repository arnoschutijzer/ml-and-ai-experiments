# bonsai-chat

A terminal chat with [Ternary Bonsai 2.5 27B](https://prismml.com/news/bonsai-2-5-27b), running locally on Apple Silicon with MLX.

## Setup

Download the MLX pack (about 8.6 GB) into `models/`:

```sh
hf download prism-ml/Ternary-Bonsai-2.5-27B-mlx-2bit --local-dir ./models/bonsai-25-mlx
```

The pack ships its own loader in `runtime/`, which applies the Hadamard transforms its weights need. The dependency versions in `pyproject.toml` match the pack's `runtime/requirements.txt`.

## Chat

```sh
uv run bonsai-chat
```

Options:

- `--no-think` starts with thinking turned off.
- `--effort xhigh|medium` sets reasoning effort. The default is `medium`.
- `--max-tokens N` caps each reply. The default is 16384.
- `--system "..."` sets the system prompt.
- `--model-dir PATH` points to a pack in another location.

In the chat, type `/help` for commands. Thinking is shown dimmed. Press Ctrl-C to stop a reply.

Sampling follows PrismML's recommendations: temperature 1.0, top-p 0.95, top-k 20 with thinking, and temperature 0.7, top-p 0.8, top-k 20, presence penalty 1.5 without.

The model needs about 9 GB of memory, plus the cache for the conversation.

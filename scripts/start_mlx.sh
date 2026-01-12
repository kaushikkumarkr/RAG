#!/bin/bash
# scripts/start_mlx.sh

MODEL_REPO="mlx-community/Qwen2.5-7B-Instruct-4bit"
PORT=8080

echo "Starting MLX Server with model: $MODEL_REPO on port $PORT"
echo "Use Ctrl+C to stop."

# Run the server module
.venv/bin/python -m mlx_lm.server --model "$MODEL_REPO" --port $PORT

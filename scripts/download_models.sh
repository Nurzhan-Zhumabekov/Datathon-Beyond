#!/bin/zsh
# Downloads the two local models (MLX 4-bit) into ./models. Run once; afterwards no internet is needed.
cd "$(dirname "$0")/.."
for MODEL in Qwen3-VL-4B-Instruct-4bit Qwen3-VL-8B-Instruct-4bit; do
  DIR="models/$MODEL"
  mkdir -p "$DIR"
  FILES=$(curl -s "https://huggingface.co/api/models/mlx-community/$MODEL/tree/main" |
    python3 -c "import sys, json; print(' '.join(f['path'] for f in json.load(sys.stdin) if f['type'] == 'file' and f['path'] not in ('README.md', '.gitattributes')))")
  for f in ${=FILES}; do
    echo "$MODEL/$f"
    curl -sSL --retry 10 --retry-all-errors -C - -o "$DIR/$f" "https://huggingface.co/mlx-community/$MODEL/resolve/main/$f"
  done
  touch "$DIR/.complete"
done
echo "Models are in ./models"

#!/bin/bash

# Usage: bash scripts/scene_stream_indoor.sh "your indoor prompt here"
# Example: bash scripts/scene_stream_indoor.sh "Create a cozy bedroom with warm lighting"

set -e

if [ $# -eq 0 ]; then
    echo "Error: Please provide user_prompt parameter"
    echo "Usage: bash scripts/scene_stream_indoor.sh \"your indoor prompt here\""
    exit 1
fi

if [ -z "${OPENAI_API_KEY:-}" ]; then
    echo "Error: OPENAI_API_KEY is not set"
    echo "Set it first, for example: export OPENAI_API_KEY=\"...\""
    exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
USER_PROMPT="$1"

cd "$REPO_ROOT"

echo "User Prompt: $USER_PROMPT"
echo ""

echo "Executing Indoor Agent 1: Task Planner"
python agent/scene_stream_indoor/planner.py "$USER_PROMPT"
echo "Indoor Task Planner completed"

echo ""
echo "Executing Indoor Agent 2: Conceptual Solver"
python agent/scene_stream_indoor/solver.py "$USER_PROMPT"
echo "Indoor Conceptual Solver completed"

echo ""
echo "Executing Indoor Agent 3: Gin Realizer"
python agent/scene_stream_indoor/realizer.py
echo "Indoor Gin Realizer completed"

CONFIGS=$(python -c "import json; print(' '.join(json.load(open('output/indoor/indoor_run_args.json', encoding='utf-8'))['configs']))")

echo ""
echo "Executing Infinigen Indoor Coarse Generation"
cd infinigen || { echo "Error: infinigen folder not found"; exit 1; }

python -m infinigen_examples.generate_indoors --seed 1 --task coarse --output_folder outputs/indoor/coarse -g $CONFIGS
echo "Indoor coarse generation completed"

echo ""
echo "Executing Infinigen Indoor Render"
python -m infinigen_examples.generate_indoors --seed 1 --task render --input_folder outputs/indoor/coarse --output_folder outputs/indoor/frames
echo "Indoor render completed"

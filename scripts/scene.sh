#!/bin/bash

# Usage:
#   bash scripts/scene.sh nature "Create a spooky forest scene with fog"
#   bash scripts/scene.sh indoor "Create a cozy bedroom with warm lighting"

set -e

if [ $# -lt 2 ]; then
    echo "Error: Please provide scene_type and user_prompt"
    echo "Usage: bash scripts/scene.sh <nature|indoor> \"your prompt here\""
    exit 1
fi

SCENE_TYPE="$1"
USER_PROMPT="$2"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

case "$SCENE_TYPE" in
    nature|outdoor)
        bash "$SCRIPT_DIR/scene_stream.sh" "$USER_PROMPT"
        ;;
    indoor)
        bash "$SCRIPT_DIR/scene_stream_indoor.sh" "$USER_PROMPT"
        ;;
    *)
        echo "Error: unknown scene_type '$SCENE_TYPE'"
        echo "Usage: bash scripts/scene.sh <nature|indoor> \"your prompt here\""
        exit 1
        ;;
esac

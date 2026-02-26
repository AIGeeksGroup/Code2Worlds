#!/bin/bash

# Usage: bash scene_stream.sh "your prompt here"
# Example: bash scene_stream.sh "Create a spooky forest scene with fog"

if [ $# -eq 0 ]; then
    echo "Error: Please provide user_prompt parameter"
    echo "Usage: bash scene_stream.sh \"your prompt here\""
    echo "Example: bash scene_stream.sh \"Create a spooky forest scene with fog\""
    exit 1
fi

USER_PROMPT="$1"

echo "User Prompt: $USER_PROMPT"
echo ""

echo "Executing Agent 1: Environment Planner"
python agent/scene_stream/planner.py "$USER_PROMPT"
if [ $? -ne 0 ]; then
    echo "Error: Planner execution failed"
    exit 1
fi
echo "Planner completed"

echo ""
echo "Executing Agent 2: Parameter Resolver"
python agent/scene_stream/resolver.py "$USER_PROMPT"
if [ $? -ne 0 ]; then
    echo "Error: Resolver execution failed"
    exit 1
fi
echo "Resolver completed"

echo ""
echo "Executing Agent 3: Scene Realizer"
python agent/scene_stream/realizer.py "$USER_PROMPT"
if [ $? -ne 0 ]; then
    echo "Error: Realizer execution failed"
    exit 1
fi
echo "Realizer completed"

echo ""
echo "Executing Infinigen Coarse Generation"
cd infinigen || { echo "Error: infinigen folder not found"; exit 1; }

python -m infinigen_examples.generate_nature --seed 1 --task coarse -g generated_scene.gin simple.gin --output_folder outputs/scene/coarse
if [ $? -ne 0 ]; then
    echo "Error: Infinigen Coarse Generation failed"
    exit 1
fi
echo "Coarse Generation completed"

echo ""
echo "Executing Infinigen Fine Generation..."
python -m infinigen_examples.generate_nature --seed 1 --task populate fine_terrain -g generated_scene.gin simple.gin --input_folder outputs/scene/coarse --output_folder outputs/scene/fine
if [ $? -ne 0 ]; then
    echo "Error: Infinigen Fine Generation failed"
    exit 1
fi
echo "Fine Generation completed"

cd ..

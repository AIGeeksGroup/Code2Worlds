#!/bin/bash

# Usage: bash scripts/obj.sh "your prompt here"
# Example: bash scripts/obj.sh "A heavy iron anvil crushing a soda can"

# Check if user_prompt parameter is provided
if [ $# -eq 0 ]; then
    echo "Error: Please provide user_prompt parameter"
    echo "Usage: bash scripts/obj.sh \"your prompt here\""
    echo "Example: bash scripts/obj.sh \"A heavy iron anvil crushing a soda can\""
    exit 1
fi

USER_PROMPT="$1"

echo "User Prompt: $USER_PROMPT"
echo ""

# Agent 1: Object Selection
echo "Executing Agent 1: Object Selection"
python agent/obj_stream/obj_select_agent.py "$USER_PROMPT"
if [ $? -ne 0 ]; then
    echo "Error: Object Selection execution failed"
    exit 1
fi
echo "Object Selection completed"

# Agent 2: Parameter Generation
echo ""
echo "Executing Agent 2: Parameter Generation"
python agent/obj_stream/obj_params_agent.py "$USER_PROMPT"
if [ $? -ne 0 ]; then
    echo "Error: Parameter Generation execution failed"
    exit 1
fi
echo "Parameter Generation completed"

# Agent 3: Code Generation
echo ""
echo "Executing Agent 3: Code Generation"
python agent/obj_stream/obj_generate_agent.py
if [ $? -ne 0 ]; then
    echo "Error: Code Generation execution failed"
    exit 1
fi
echo "Code Generation completed"

# Switch to infinigen directory and execute generated code
echo ""
echo "Switching to infinigen directory and executing generated code"
cd infinigen || { echo "Error: infinigen folder not found"; exit 1; }

python obj_code.py
if [ $? -ne 0 ]; then
    echo "Error: obj_code.py execution failed"
    exit 1
fi
echo "obj_code.py execution completed"

# Return to original directory
cd ..

# Refinement Loop: Iterate until validation passes or max iterations reached
MAX_ITERATIONS=5
FEEDBACK_FILE="./output/obj/reflection_feedback.json"

for iteration in $(seq 1 $MAX_ITERATIONS); do
    echo ""
    echo "Iteration $iteration/$MAX_ITERATIONS"

    
    # Execute rendering
    echo ""
    echo "Executing Object Rendering"
    python agent/obj_stream/render_object.py
    if [ $? -ne 0 ]; then
        echo "Error: Object Rendering execution failed"
        exit 1
    fi
    echo "Object Rendering completed"
    
    # Agent 4: Reflection (Visual Evaluation)
    echo ""
    echo "Executing Agent 4: Reflection"
    python agent/obj_stream/objreflection.py "$USER_PROMPT"
    if [ $? -ne 0 ]; then
        echo "Error: Reflection execution failed"
        exit 1
    fi
    echo "Reflection completed"
    
    # Check validation result
    if [ -f "$FEEDBACK_FILE" ]; then
        IS_VALID=$(python -c "import json; data=json.load(open('$FEEDBACK_FILE')); print(data.get('valid', False))")
        
        if [ "$IS_VALID" = "True" ]; then
            echo ""
            echo "Validation passed! Object generation successful!"
            break
        else
            echo ""
            echo "Validation failed. Feedback:"
            python -c "import json; data=json.load(open('$FEEDBACK_FILE')); print('  ', data.get('feedback', 'No feedback'))"
            
            if [ $iteration -lt $MAX_ITERATIONS ]; then
                echo ""
                echo "Regenerating based on feedback... (Attempt $iteration/$MAX_ITERATIONS)"
                
                # Re-run Agent 2: Parameter Generation with feedback
                echo ""
                echo "Re-executing Agent 2: Parameter Generation"
                python agent/obj_stream/obj_params_agent.py "$USER_PROMPT"
                if [ $? -ne 0 ]; then
                    echo "Error: Parameter Generation execution failed"
                    exit 1
                fi
                echo "Parameter Generation completed"
                
                # Re-run Agent 3: Code Generation
                echo ""
                echo "Re-executing Agent 3: Code Generation"
                python agent/obj_stream/obj_generate_agent.py
                if [ $? -ne 0 ]; then
                    echo "Error: Code Generation execution failed"
                    exit 1
                fi
                echo "Code Generation completed"
                
                # Re-execute generated code in infinigen
                echo ""
                echo "Re-executing generated code in infinigen directory"
                cd infinigen || { echo "Error: infinigen folder not found"; exit 1; }
                python obj_code.py
                if [ $? -ne 0 ]; then
                    echo "Error: obj_code.py execution failed"
                    exit 1
                fi
                echo "obj_code.py execution completed"
                cd ..
            else
                echo ""
                echo "Maximum iterations reached ($MAX_ITERATIONS), but validation still failed."
            fi
        fi
    else
        echo "Warning: Feedback file not found: $FEEDBACK_FILE"
        break
    fi
done
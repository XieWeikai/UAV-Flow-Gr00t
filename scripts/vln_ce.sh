#!/bin/bash

# Configuration
OUTPUT_DIR="/data-10T/InternData-N1"
NUM_PROCESSES=16

# Directories to process
# Format: "Path"
# The script uses the directory name (basename) as the task ID.
RAW_DIRS=(
    "/data-10T/InternData-N1/r2r"
    "/data-10T/InternData-N1/rxr"
    "/data-10T/InternData-N1/scalevln"
)

# -----------------------------------------------------------------------------
# Helper function to print usage
usage() {
    echo "Usage: $0 [task_name...]"
    echo ""
    echo "Arguments:"
    echo "  task_name   (Optional) Process only specific datasets matching the folder name."
    echo ""
    echo "Available tasks (based on configured directories):"
    for dir in "${RAW_DIRS[@]}"; do
        echo "  - $(basename "$dir")"
    done
    echo ""
    echo "Example:"
    echo "  $0                # Run all tasks"
    echo "  $0 r2r            # Run only the 'r2r' task"
    echo "  $0 r2r scalevln   # Run 'r2r' and 'scalevln'"
}

# Check for help flag
if [[ "$1" == "-h" || "$1" == "--help" ]]; then
    usage
    exit 0
fi

# Filter commands to run
TARGET_DIRS=()
if [ $# -eq 0 ]; then
    # No args, run all
    TARGET_DIRS=("${RAW_DIRS[@]}")
else
    # Parse args and match against basenames
    for arg in "$@"; do
        MATCHED=false
        for dir in "${RAW_DIRS[@]}"; do
            NAME=$(basename "$dir")
            if [ "$NAME" == "$arg" ]; then
                TARGET_DIRS+=("$dir")
                MATCHED=true
                break
            fi
        done
        if [ "$MATCHED" = false ]; then
            echo "Error: Task '$arg' not found in configuration."
            echo "Available tasks: $(printf '%s ' $(for d in "${RAW_DIRS[@]}"; do basename "$d"; done))"
            exit 1
        fi
    done
fi

echo "Starting execution for: $(for d in "${TARGET_DIRS[@]}"; do basename "$d"; done)"

# Create a logs directory
mkdir -p logs

PIDS=()
NAMES=()

for dir in "${TARGET_DIRS[@]}"; do
    NAME=$(basename "$dir")
    LOG_FILE="logs/${NAME}.log"
    echo "[$NAME] Running... (Log: $LOG_FILE)"
    
    # Run in background
    # Redirect both stdout and stderr to log file for cleaner parallel execution
    uv run vln_ce.py --raw_dir "$dir" --output_dir "$OUTPUT_DIR" --num_processes "$NUM_PROCESSES" > "$LOG_FILE" 2>&1 &
    
    PID=$!
    PIDS+=($PID)
    NAMES+=("$NAME")
done

# Wait for all processes and check status
FAILED=0
for i in "${!PIDS[@]}"; do
    PID=${PIDS[$i]}
    NAME=${NAMES[$i]}
    
    wait $PID
    STATUS=$?
    
    if [ $STATUS -eq 0 ]; then
        echo "[$NAME] Finished successfully."
    else
        echo "[$NAME] FAILED with status $STATUS. Check logs/${NAME}.log"
        FAILED=1
    fi
done

if [ $FAILED -ne 0 ]; then
    echo "Some tasks failed."
    exit 1
else
    echo "All tasks finished successfully."
fi


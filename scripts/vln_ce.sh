#!/bin/bash
set -o pipefail  # Ensure pipeline errors are caught

# Configuration
OUTPUT_DIR="/data-10T/InternData-N1"
NUM_PROCESSES=16
MAX_LOG_LINES=2000

# Directories to process
# Format: "Path"
# The script uses the directory name (basename) as the task ID.
RAW_DIRS=(
    "/data-10T/InternData-N1/r2r"
    "/data-10T/InternData-N1/rxr"
    "/data-10T/InternData-N1/scalevln"
)

# -----------------------------------------------------------------------------
# Function: Real-time log limiter
# Reads from stdin and maintains a file with at most MAX_LINES.
# Optimization: Appends continuously, and only performs a truncate/rewrite
# cycle when the file grows significantly beyond the limit (e.g. +20%).
limit_log_output() {
    local log_file="$1"
    local max_lines="$2"

    python3 -u -c "
import sys
import os

log_file = '$log_file'
max_lines = $max_lines
threshold = int(max_lines * 1.2)  # Truncate when we exceed 20% over limit

current_lines = 0

# Ensure file exists
with open(log_file, 'w') as f:
    pass

try:
    with open(log_file, 'a') as f:
        for line in sys.stdin:
            f.write(line)
            f.flush()
            current_lines += 1
            
            # Periodically truncate the file to keep it small
            if current_lines > threshold:
                # Read current content
                with open(log_file, 'r') as fr:
                    content = fr.readlines()
                
                # Keep last max_lines
                new_content = content[-max_lines:]
                
                # Rewrite file
                with open(log_file, 'w') as fw:
                    fw.writelines(new_content)
                    fw.flush() # Ensure update is visible
                
                # Re-open in append mode for next loop
                # Note: 'f' is still open but pointing to old file handle technically, 
                # but since we are in a loop with 'with open', we need to be careful.
                # Actually, rewriting the file underneath an open file handle 'f' 
                # might cause issues on some OS, but on Linux 'w' creates a new inode 
                # or truncates. Let's close and reopen properly conceptually, 
                # but inside this loop 'f' is fixed.
                
                # Better approach for the loop:
                current_lines = len(new_content)

    # To fix the file handle issue above, we can't easily swap 'f' inside the loop.
    # Simpler logic for correctness:
except BrokenPipeError:
    pass
" 
    # Fallback to a simpler but slower approach if the above is too complex to inline reliably:
    # Pure Python Re-writer (Writes every line)
    # This is IO heavy but 100% correct and simpler to inline.
    python3 -u -c "
import sys
from collections import deque

log_file = '$log_file'
max_lines = $max_lines
q = deque(maxlen=max_lines)

# Initialize
with open(log_file, 'w') as f: pass

try:
    for line in sys.stdin:
        q.append(line)
        # Optimization: batch write every 10 lines or just write every line
        # Writing every line ensures 'tail -f' works roughly well
        with open(log_file, 'w') as f:
            f.writelines(q)
except (BrokenPipeError, KeyboardInterrupt):
    pass
"
}

# Use the cleaner optimization logic in a function
run_with_log_limit() {
    local cmd="$1"
    local log_file="$2"
    
    # Run command, pipe to python script that manages the file
    # We use a python script that appends and truncates cleanly
    eval "$cmd" 2>&1 | python3 -u -c "
import sys
import collections

filepath = '$log_file'
limit = $MAX_LOG_LINES
# Buffer size before enforcing limit (reduce IO frequency)
flush_interval = 20 
buffer = []

with open(filepath, 'w') as f:
    pass

try:
    lines_since_flush = 0
    # We keep the whole history in memory? No, that defeats the purpose.
    # We maintain a deque in memory and fully rewrite the file periodically.
    # This is the most stable way to ensure the file on disk is small.
    
    memory_buffer = collections.deque(maxlen=limit)
    
    for line in sys.stdin:
        # Also print to stdout (terminal)
        sys.stdout.write(line)
        sys.stdout.flush()

        memory_buffer.append(line)
        lines_since_flush += 1

        
        if lines_since_flush >= flush_interval:
            with open(filepath, 'w') as f:
                f.writelines(memory_buffer)
            lines_since_flush = 0
            
    # Final flush
    with open(filepath, 'w') as f:
        f.writelines(memory_buffer)
        
except (BrokenPipeError, KeyboardInterrupt):
    pass
"
}


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
    echo "[$NAME] Running... (Log: $LOG_FILE, Max Lines: $MAX_LOG_LINES)"
    
    # Run using the wrapper function for log limiting
    run_with_log_limit "uv run vln_ce.py --raw_dir '$dir' --output_dir '$OUTPUT_DIR' --num_processes '$NUM_PROCESSES'" "$LOG_FILE" &
    
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


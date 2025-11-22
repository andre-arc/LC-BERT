#!/bin/bash
# Wrapper script to run training tasks with Telegram notifications
# Usage: ./run_with_notification.sh <script_to_run> [args...]
# Example: ./run_with_notification.sh run_task_modified.sh 0 3 32

# Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
# Get the project root directory (parent of linux_scripts)
PROJECT_DIR="$( cd "$SCRIPT_DIR/.." && pwd )"

# Check if a script was provided
if [ -z "$1" ]; then
    echo "Error: No script specified"
    echo "Usage: ./run_with_notification.sh <script_to_run> [args...]"
    echo "Example: ./run_with_notification.sh run_task_modified.sh 0 3 32"
    exit 1
fi

# Store the script name and all arguments
SCRIPT_NAME="$1"
shift
SCRIPT_ARGS="$@"

# Check if Telegram credentials are set
if [ -z "$TELEGRAM_BOT_TOKEN" ]; then
    echo "Warning: TELEGRAM_BOT_TOKEN environment variable not set"
    echo "Notifications will be skipped"
    NOTIFY_ENABLED=0
elif [ -z "$TELEGRAM_CHAT_ID" ]; then
    echo "Warning: TELEGRAM_CHAT_ID environment variable not set"
    echo "Notifications will be skipped"
    NOTIFY_ENABLED=0
else
    NOTIFY_ENABLED=1
fi

# Extract experiment name from arguments if present
EXPERIMENT_NAME="unknown"
if echo "$SCRIPT_ARGS" | grep -q -- "--experiment_name"; then
    EXPERIMENT_NAME=$(echo "$SCRIPT_ARGS" | grep -oP -- '--experiment_name\s+\K[^\s]+' | head -1)
fi

# Setup flag file path (using temp directory and sanitized task name)
FLAG_DIR="/tmp/lc-bert-notify"
mkdir -p "$FLAG_DIR"
SANITIZED_NAME=$(echo "$SCRIPT_NAME" | tr ' :.' '_--' | sed 's/\.sh$//')
FLAG_FILE="$FLAG_DIR/$SANITIZED_NAME.running"

# Check for orphaned flag file from previous interrupted run
if [ -f "$FLAG_FILE" ]; then
    if [ $NOTIFY_ENABLED -eq 1 ]; then
        echo "Found orphaned task from previous run, sending interrupted notification..."
        python "$PROJECT_DIR/telegram_notifier.py" --status interrupted --task-name "$SCRIPT_NAME" --experiment-name "$EXPERIMENT_NAME" --details "Previous run was interrupted or terminated unexpectedly"
    fi
    rm -f "$FLAG_FILE"
fi

# Create flag file to track running task
echo "$SCRIPT_NAME" > "$FLAG_FILE"

# Send start notification
if [ $NOTIFY_ENABLED -eq 1 ]; then
    echo "Sending start notification..."
    python "$PROJECT_DIR/telegram_notifier.py" --status started --task-name "$SCRIPT_NAME" --experiment-name "$EXPERIMENT_NAME" --details "Arguments: $SCRIPT_ARGS"
fi

# Run the actual script
echo ""
echo "================================================"
echo "Running: $SCRIPT_NAME $SCRIPT_ARGS"
echo "================================================"
echo ""

bash "$SCRIPT_DIR/$SCRIPT_NAME" $SCRIPT_ARGS

# Capture exit code
EXIT_CODE=$?

# Send completion notification
if [ $NOTIFY_ENABLED -eq 1 ]; then
    if [ $EXIT_CODE -eq 0 ]; then
        echo ""
        echo "Sending success notification..."
        python "$PROJECT_DIR/telegram_notifier.py" --status success --task-name "$SCRIPT_NAME" --experiment-name "$EXPERIMENT_NAME" --details "Task completed successfully"
    else
        echo ""
        echo "Sending failure notification..."
        python "$PROJECT_DIR/telegram_notifier.py" --status failed --task-name "$SCRIPT_NAME" --experiment-name "$EXPERIMENT_NAME" --details "Task failed with exit code: $EXIT_CODE"
    fi
fi

# Clean up flag file on normal completion or failure
rm -f "$FLAG_FILE"

# Exit with the original exit code
exit $EXIT_CODE

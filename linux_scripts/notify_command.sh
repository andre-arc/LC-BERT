#!/bin/bash
# Helper script to run a single command with Telegram notifications
# Usage: ./notify_command.sh <experiment_name> <command>
# Example: ./notify_command.sh "ZCA-BiLSTM" python main.py --experiment_name ag-news-bert-whitening-zca-bilstm ...

# Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
# Get the project root directory (parent of linux_scripts)
PROJECT_DIR="$( cd "$SCRIPT_DIR/.." && pwd )"

# Check if arguments were provided
if [ -z "$1" ]; then
    echo "Error: No experiment name specified"
    echo "Usage: ./notify_command.sh <experiment_name> <command>"
    exit 1
fi

if [ -z "$2" ]; then
    echo "Error: No command specified"
    echo "Usage: ./notify_command.sh <experiment_name> <command>"
    exit 1
fi

# Extract experiment name and build full command
EXPERIMENT_NAME="$1"
shift
FULL_COMMAND="$@"

# Check if Telegram credentials are set
NOTIFY_ENABLED=0
if [ -n "$TELEGRAM_BOT_TOKEN" ] && [ -n "$TELEGRAM_CHAT_ID" ]; then
    NOTIFY_ENABLED=1
fi

# Setup flag file path (using temp directory and sanitized experiment name)
FLAG_DIR="/tmp/lc-bert-notify"
mkdir -p "$FLAG_DIR"
SANITIZED_NAME=$(echo "$EXPERIMENT_NAME" | tr ' :' '_-')
FLAG_FILE="$FLAG_DIR/$SANITIZED_NAME.running"

# Check for orphaned flag file from previous interrupted run
if [ -f "$FLAG_FILE" ]; then
    if [ $NOTIFY_ENABLED -eq 1 ]; then
        echo "[NOTIFY] Found orphaned task, sending interrupted notification"
        python "$PROJECT_DIR/telegram_notifier.py" --status interrupted --task-name "$EXPERIMENT_NAME" --details "Previous run was interrupted or terminated unexpectedly"
    fi
    rm -f "$FLAG_FILE"
fi

# Create flag file to track running task
echo "$EXPERIMENT_NAME" > "$FLAG_FILE"

# Send start notification
if [ $NOTIFY_ENABLED -eq 1 ]; then
    echo "[NOTIFY] Starting: $EXPERIMENT_NAME"
    python "$PROJECT_DIR/telegram_notifier.py" --status started --task-name "$EXPERIMENT_NAME" --details "Command: $FULL_COMMAND"
fi

# Run the command
echo ""
echo "================================================"
echo "Running: $EXPERIMENT_NAME"
echo "================================================"
$FULL_COMMAND

# Capture exit code
EXIT_CODE=$?

# Send completion notification
if [ $NOTIFY_ENABLED -eq 1 ]; then
    if [ $EXIT_CODE -eq 0 ]; then
        echo "[NOTIFY] Success: $EXPERIMENT_NAME"
        python "$PROJECT_DIR/telegram_notifier.py" --status success --task-name "$EXPERIMENT_NAME" --details "Training completed successfully"
    else
        echo "[NOTIFY] Failed: $EXPERIMENT_NAME"
        python "$PROJECT_DIR/telegram_notifier.py" --status failed --task-name "$EXPERIMENT_NAME" --details "Training failed with exit code: $EXIT_CODE"
    fi
fi

# Clean up flag file on normal completion or failure
rm -f "$FLAG_FILE"

# Exit with the original exit code
exit $EXIT_CODE

#!/bin/bash
# Telegram Bot Configuration
# This script reads configuration from .env file in the parent directory
#
# To get these values, follow the instructions in TELEGRAM_SETUP.md
#
# IMPORTANT: Never commit .env to Git!
# It's already added to .gitignore for your safety.

# Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Set the path to the .env file (in parent directory)
ENV_FILE="$SCRIPT_DIR/../.env"

# Check if .env file exists
if [ ! -f "$ENV_FILE" ]; then
    echo "Error: .env file not found at $ENV_FILE"
    echo "Please create a .env file with your Telegram credentials."
    echo "See .env.example for the required format."
    return 1 2>/dev/null || exit 1
fi

# Read the .env file and export environment variables
# Skip empty lines and comments
while IFS='=' read -r key value; do
    # Skip empty lines
    if [ -z "$key" ]; then
        continue
    fi

    # Skip comments
    if [[ "$key" =~ ^[[:space:]]*# ]]; then
        continue
    fi

    # Remove leading/trailing whitespace and quotes
    key=$(echo "$key" | sed -e 's/^[[:space:]]*//' -e 's/[[:space:]]*$//')
    value=$(echo "$value" | sed -e 's/^[[:space:]]*//' -e 's/[[:space:]]*$//' -e 's/^"//' -e 's/"$//' -e "s/^'//" -e "s/'$//")

    # Export the variable
    export "$key=$value"
done < "$ENV_FILE"

# Verify required variables are set
if [ -z "$TELEGRAM_BOT_TOKEN" ]; then
    echo "Error: TELEGRAM_BOT_TOKEN not found in .env file"
    return 1 2>/dev/null || exit 1
fi

if [ -z "$TELEGRAM_CHAT_ID" ]; then
    echo "Error: TELEGRAM_CHAT_ID not found in .env file"
    return 1 2>/dev/null || exit 1
fi

# Display configuration (masked for security)
echo ""
echo "Telegram Bot Configuration loaded:"
echo "- Bot Token: ${TELEGRAM_BOT_TOKEN:0:10}..."
echo "- Chat ID: $TELEGRAM_CHAT_ID"
if [ -n "$TELEGRAM_MESSAGE_THREAD_ID" ]; then
    echo "- Thread ID: $TELEGRAM_MESSAGE_THREAD_ID"
fi
echo ""

#!/bin/bash
# Telegram Bot Configuration Template
# 
# Instructions:
# 1. Copy this file: cp telegram_config.template.sh telegram_config.sh
# 2. Edit telegram_config.sh with your actual credentials
# 3. Source it before running: source telegram_config.sh
#
# To get these values:
# 1. Talk to @BotFather on Telegram to create a bot and get the token
# 2. Get your chat ID by messaging @userinfobot or your bot
# 3. For group chats, add the bot to the group and get the group chat ID
# 4. For forum topics, get the message thread ID from the topic
#
# IMPORTANT: 
# - Never commit telegram_config.sh to Git!
# - Add it to .gitignore: echo "telegram_config.sh" >> .gitignore
# - Only commit this template file

# Your Telegram Bot Token (from @BotFather)
# Example: 123456789:ABCdefGHIjklMNOpqrsTUVwxyz
export TELEGRAM_BOT_TOKEN="YOUR_BOT_TOKEN_HERE"

# Your Telegram Chat ID (personal chat or group chat)
# Example: 123456789 (for personal) or -987654321 (for group)
export TELEGRAM_CHAT_ID="YOUR_CHAT_ID_HERE"

# Your Telegram Message Thread ID (for forum topics in groups)
# Leave empty if not using forum topics
# Example: 4
export TELEGRAM_MESSAGE_THREAD_ID=""

# Optional: Disable notifications temporarily
# Uncomment to disable even if credentials are set
# unset TELEGRAM_BOT_TOKEN
# unset TELEGRAM_CHAT_ID

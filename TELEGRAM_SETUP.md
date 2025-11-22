# Telegram Notification Setup Guide

This guide explains how to set up Telegram notifications for your training tasks.

## Prerequisites

- Python 3.x installed
- `requests` library installed: `pip install requests`
- A Telegram account

## Step 1: Create a Telegram Bot

1. Open Telegram and search for `@BotFather`
2. Start a chat with BotFather and send `/newbot`
3. Follow the instructions to create your bot:
   - Choose a name for your bot (e.g., "LC-BERT Training Monitor")
   - Choose a username for your bot (must end in 'bot', e.g., "lcbert_monitor_bot")
4. BotFather will give you a **bot token**. Save this token securely.
   - Example: `123456789:ABCdefGHIjklMNOpqrsTUVwxyz`

## Step 2: Get Your Chat ID

### For Personal Chat (Direct Messages):

1. Search for your bot's username in Telegram and start a chat
2. Send any message to your bot (e.g., "/start")
3. Open this URL in your browser (replace `YOUR_BOT_TOKEN` with your actual token):
   ```
   https://api.telegram.org/botYOUR_BOT_TOKEN/getUpdates
   ```
4. Look for `"chat":{"id":` in the response. The number after it is your chat ID.
   - Example: `"chat":{"id":123456789`
   - Your chat ID: `123456789`

### For Group Chat:

1. Create a new Telegram group (or use an existing one)
2. Add your bot to the group as a member
3. Send a message in the group (e.g., "Hello bot!")
4. Open this URL in your browser (replace `YOUR_BOT_TOKEN` with your actual token):
   ```
   https://api.telegram.org/botYOUR_BOT_TOKEN/getUpdates
   ```
5. Look for the group chat in the response. The chat ID will be a negative number.
   - Example: `"chat":{"id":-987654321`
   - Your chat ID: `-987654321`

## Step 3: Set Environment Variables

### Option A: Set Permanently (Windows)

1. Open System Properties (Win + Pause/Break or search "Environment Variables")
2. Click "Environment Variables"
3. Under "User variables" or "System variables", click "New"
4. Add these two variables:
   - Variable name: `TELEGRAM_BOT_TOKEN`
   - Variable value: Your bot token

   - Variable name: `TELEGRAM_CHAT_ID`
   - Variable value: Your chat ID
5. Click OK and restart any open command prompts

### Option B: Set in Current Session (Windows)

```batch
set TELEGRAM_BOT_TOKEN=123456789:ABCdefGHIjklMNOpqrsTUVwxyz
set TELEGRAM_CHAT_ID=123456789
```

### Option C: Set in Batch File

Create a file named `telegram_config.bat`:
```batch
@echo off
set TELEGRAM_BOT_TOKEN=123456789:ABCdefGHIjklMNOpqrsTUVwxyz
set TELEGRAM_CHAT_ID=123456789
```

Then call it before running your tasks:
```batch
call telegram_config.bat
run_with_notification.bat run_task_modified.bat 0 3 32
```

**⚠️ Security Note**: Never commit `telegram_config.bat` to Git if it contains your real tokens!

## Step 4: Install Required Python Package

```bash
pip install requests
```

Or add to your `requirements.txt` (if not already there).

## Step 5: Test the Notification

Test if everything works:

```batch
python telegram_notifier.py --status success --task-name "Test Task" --details "This is a test notification"
```

If successful, you should receive a message in your Telegram chat/group!

## Usage

### Method 1: Using the Wrapper Script (Recommended)

Run any batch file with automatic notifications:

```batch
# Standard training with notifications
run_with_notification.bat run_task_modified.bat 0 3 32

# K-fold training with notifications
run_with_notification.bat run_task_modified_kfold.bat 0 3 32
```

The wrapper will automatically:
- Send a "STARTED" notification when the task begins
- Send a "SUCCESS" notification if the task completes successfully
- Send a "FAILED" notification if the task fails

### Method 2: Manual Notifications in Your Scripts

Add notification calls directly to your batch files:

```batch
@echo off

REM Send start notification
python telegram_notifier.py --status started --task-name "Training" --experiment-name "my-experiment"

REM Your training command
python main.py --n_epochs 5 --experiment_name my-experiment [other args...]

REM Check if successful
if %errorlevel% equ 0 (
    python telegram_notifier.py --status success --task-name "Training" --experiment-name "my-experiment"
) else (
    python telegram_notifier.py --status failed --task-name "Training" --experiment-name "my-experiment"
)
```

### Method 3: Direct Python Call

Call from within your Python scripts:

```python
import os
os.system('python telegram_notifier.py --status success --task-name "My Task" --details "Training complete"')
```

## Notification Format

Notifications include:
- **Status** (with emoji): 🚀 STARTED, ✅ SUCCESS, ❌ FAILED
- **Task name**: The batch file or task being run
- **Experiment name**: The experiment being trained (if provided)
- **Machine**: The hostname of the machine running the task
- **Timestamp**: When the notification was sent
- **Details**: Additional information about the task

Example notification:
```
✅ SUCCESS
━━━━━━━━━━━━━━━━━━━━
Task: run_task_modified.bat
Experiment: ag-news-bert-whitening-zca-bilstm
Machine: DESKTOP-ABC123
Time: 2025-11-12 14:30:45

Details:
Task completed successfully
```

## Troubleshooting

### "Failed to send Telegram notification"

1. Check your internet connection
2. Verify your bot token is correct
3. Verify your chat ID is correct
4. Make sure you've started a chat with your bot (for personal chats)
5. Make sure the bot is still a member of your group (for groups)

### "TELEGRAM_BOT_TOKEN environment variable not set"

1. Make sure you set the environment variables (Step 3)
2. Restart your command prompt after setting environment variables
3. Check spelling: `TELEGRAM_BOT_TOKEN` and `TELEGRAM_CHAT_ID`

### Bot doesn't send messages to the group

1. Make sure the bot is added to the group
2. Make sure you're using the correct (negative) group chat ID
3. Try sending a test message: `/start` in the group

### Python module not found

```bash
pip install requests
```

## Advanced: Using MCP Servers (Optional)

If you want to integrate with Claude Code's MCP (Model Context Protocol) capabilities, you could create an MCP server that provides notification functionality. However, the direct Python approach above is simpler and more straightforward for this use case.

## Security Best Practices

1. **Never commit tokens to Git**: Add `telegram_config.bat` to `.gitignore`
2. **Regenerate compromised tokens**: If you accidentally expose your token, use BotFather to regenerate it
3. **Use environment variables**: Store sensitive data in environment variables, not in code
4. **Limit bot permissions**: Telegram bots only have the permissions you grant them

## Integration with Existing Scripts

To integrate with your existing training scripts without modifying them:

```batch
# Add this to the top of run_task_modified.bat
@echo off
REM Check if this script is being run through the wrapper
if not "%NOTIFICATION_WRAPPER%"=="1" (
    set NOTIFICATION_WRAPPER=1
    run_with_notification.bat %~f0 %*
    exit /b
)

REM Original script content continues here...
set CUDA_VISIBLE_DEVICES=%1
python main.py [args...]
```

This makes the script automatically use notifications when run directly!

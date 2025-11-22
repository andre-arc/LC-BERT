"""
Telegram Notification Script for LC-BERT Training Tasks

This script sends notifications to a Telegram group/chat about task status.
"""

import os
import sys
import requests
from datetime import datetime
import argparse
import socket

def send_telegram_message(bot_token, chat_id, message, parse_mode="HTML", message_thread_id=None):
    """
    Send a message to a Telegram chat.

    Args:
        bot_token (str): Telegram bot token
        chat_id (str): Telegram chat ID (can be group ID)
        message (str): Message to send
        parse_mode (str): Message formatting mode (HTML, Markdown, or None)
        message_thread_id (int, optional): Message thread ID for forum topics

    Returns:
        bool: True if message sent successfully, False otherwise
    """
    url = f"https://api.telegram.org/bot{bot_token}/sendMessage"

    payload = {
        "chat_id": chat_id,
        "text": message,
    }

    if parse_mode:
        payload["parse_mode"] = parse_mode

    if message_thread_id:
        payload["message_thread_id"] = int(message_thread_id)

    try:
        response = requests.post(url, json=payload, timeout=10)
        response.raise_for_status()
        return True
    except requests.exceptions.RequestException as e:
        print(f"Failed to send Telegram message: {e}")
        return False

def format_task_notification(task_name, status, details=None, experiment_name=None):
    """
    Format a notification message for a task.

    Args:
        task_name (str): Name of the task
        status (str): Status of the task ('started', 'success', 'failed', 'interrupted')
        details (str, optional): Additional details about the task
        experiment_name (str, optional): Name of the experiment

    Returns:
        str: Formatted message
    """
    hostname = socket.gethostname()
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Status emoji and formatting
    status_info = {
        'started': {'emoji': '🚀', 'text': 'STARTED'},
        'success': {'emoji': '✅', 'text': 'SUCCESS'},
        'failed': {'emoji': '❌', 'text': 'FAILED'},
        'interrupted': {'emoji': '⚠️', 'text': 'INTERRUPTED'}
    }

    info = status_info.get(status.lower(), {'emoji': 'ℹ️', 'text': status.upper()})

    message = f"{info['emoji']} <b>{info['text']}</b>\n"
    message += f"━━━━━━━━━━━━━━━━━━━━\n"
    message += f"<b>Task:</b> {task_name}\n"

    if experiment_name:
        message += f"<b>Experiment:</b> {experiment_name}\n"

    message += f"<b>Machine:</b> {hostname}\n"
    message += f"<b>Time:</b> {timestamp}\n"

    if details:
        message += f"\n<b>Details:</b>\n{details}"

    return message

def main():
    parser = argparse.ArgumentParser(description='Send Telegram notifications for training tasks')
    parser.add_argument('--status', required=True, choices=['started', 'success', 'failed', 'interrupted'],
                        help='Task status')
    parser.add_argument('--task-name', required=True, help='Name of the task')
    parser.add_argument('--experiment-name', help='Name of the experiment')
    parser.add_argument('--details', help='Additional details about the task')
    parser.add_argument('--bot-token', help='Telegram bot token (or set TELEGRAM_BOT_TOKEN env var)')
    parser.add_argument('--chat-id', help='Telegram chat ID (or set TELEGRAM_CHAT_ID env var)')
    parser.add_argument('--thread-id', help='Telegram message thread ID for forum topics (or set TELEGRAM_MESSAGE_THREAD_ID env var)')

    args = parser.parse_args()

    # Get credentials from args or environment variables
    bot_token = args.bot_token or os.environ.get('TELEGRAM_BOT_TOKEN')
    chat_id = args.chat_id or os.environ.get('TELEGRAM_CHAT_ID')
    thread_id = args.thread_id or os.environ.get('TELEGRAM_MESSAGE_THREAD_ID')

    if not bot_token or not chat_id:
        print("Error: Telegram bot token and chat ID must be provided either as arguments or environment variables")
        print("Set TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID environment variables, or use --bot-token and --chat-id")
        sys.exit(1)

    # Format and send message
    message = format_task_notification(
        task_name=args.task_name,
        status=args.status,
        details=args.details,
        experiment_name=args.experiment_name
    )

    success = send_telegram_message(bot_token, chat_id, message, message_thread_id=thread_id)

    if success:
        print("Telegram notification sent successfully")
        sys.exit(0)
    else:
        print("Failed to send Telegram notification")
        sys.exit(1)

if __name__ == "__main__":
    main()

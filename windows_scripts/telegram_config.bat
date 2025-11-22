@echo off
REM Telegram Bot Configuration
REM This script reads configuration from .env file
REM
REM To get these values, follow the instructions in TELEGRAM_SETUP.md
REM
REM IMPORTANT: Never commit .env to Git!
REM It's already added to .gitignore for your safety.

setlocal enabledelayedexpansion

REM Set the path to the .env file (in parent directory)
set "ENV_FILE=%~dp0..\.env"

REM Check if .env file exists
if not exist "%ENV_FILE%" (
    echo Error: .env file not found at %ENV_FILE%
    echo Please create a .env file with your Telegram credentials.
    echo See .env.example for the required format.
    exit /b 1
)

REM Read the .env file and set environment variables
for /f "usebackq tokens=1,* delims==" %%a in ("%ENV_FILE%") do (
    set "line=%%a"
    REM Skip empty lines and comments
    if not "!line!"=="" (
        if not "!line:~0,1!"=="#" (
            REM Remove any leading/trailing spaces and quotes
            set "key=%%a"
            set "value=%%b"
            REM Remove quotes from value if present
            set "value=!value:"=!"
            REM Set the environment variable
            set "!key!=!value!"
        )
    )
)

REM Verify required variables are set
if not defined TELEGRAM_BOT_TOKEN (
    echo Error: TELEGRAM_BOT_TOKEN not found in .env file
    exit /b 1
)

if not defined TELEGRAM_CHAT_ID (
    echo Error: TELEGRAM_CHAT_ID not found in .env file
    exit /b 1
)

REM TELEGRAM_MESSAGE_THREAD_ID is optional
if not defined TELEGRAM_MESSAGE_THREAD_ID (
    set "TELEGRAM_MESSAGE_THREAD_ID="
)

REM Display configuration (masked for security)
echo.
echo Telegram Bot Configuration loaded:
echo - Bot Token: %TELEGRAM_BOT_TOKEN:~0,10%...
echo - Chat ID: %TELEGRAM_CHAT_ID%
if defined TELEGRAM_MESSAGE_THREAD_ID (
    echo - Thread ID: %TELEGRAM_MESSAGE_THREAD_ID%
)
echo.

endlocal & (
    set "TELEGRAM_BOT_TOKEN=%TELEGRAM_BOT_TOKEN%"
    set "TELEGRAM_CHAT_ID=%TELEGRAM_CHAT_ID%"
    set "TELEGRAM_MESSAGE_THREAD_ID=%TELEGRAM_MESSAGE_THREAD_ID%"
)
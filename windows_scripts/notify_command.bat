@echo off
REM Helper script to run a single command with Telegram notifications
REM Usage: call notify_command.bat <experiment_name> <command>
REM Example: call notify_command.bat "ZCA-BiLSTM" python main.py --experiment_name ag-news-bert-whitening-zca-bilstm ...

setlocal enabledelayedexpansion

REM Check if arguments were provided
if "%~1"=="" (
    echo Error: No experiment name specified
    echo Usage: call notify_command.bat ^<experiment_name^> ^<command^>
    exit /b 1
)

if "%~2"=="" (
    echo Error: No command specified
    echo Usage: call notify_command.bat ^<experiment_name^> ^<command^>
    exit /b 1
)

REM Extract experiment name and build full command
set "EXPERIMENT_NAME=%~1"
set "FULL_COMMAND=%~2"
shift
shift

:build_command
if "%~1"=="" goto done_building
set "FULL_COMMAND=!FULL_COMMAND! %~1"
shift
goto build_command

:done_building

REM Check if Telegram credentials are set
set "NOTIFY_ENABLED=0"
if not "%TELEGRAM_BOT_TOKEN%"=="" (
    if not "%TELEGRAM_CHAT_ID%"=="" (
        set "NOTIFY_ENABLED=1"
    )
)

REM Setup flag file path (using temp directory and sanitized experiment name)
set "FLAG_DIR=%TEMP%\lc-bert-notify"
if not exist "!FLAG_DIR!" mkdir "!FLAG_DIR!"
set "SANITIZED_NAME=%EXPERIMENT_NAME: =_%"
set "SANITIZED_NAME=!SANITIZED_NAME::=-!"
set "FLAG_FILE=!FLAG_DIR!\!SANITIZED_NAME!.running"

REM Check for orphaned flag file from previous interrupted run
if exist "!FLAG_FILE!" (
    if "!NOTIFY_ENABLED!"=="1" (
        echo [NOTIFY] Found orphaned task, sending interrupted notification
        python telegram_notifier.py --status interrupted --task-name "%EXPERIMENT_NAME%" --details "Previous run was interrupted or terminated unexpectedly"
    )
    del "!FLAG_FILE!" 2>nul
)

REM Create flag file to track running task
echo %EXPERIMENT_NAME% > "!FLAG_FILE!"

REM Send start notification
if "!NOTIFY_ENABLED!"=="1" (
    echo [NOTIFY] Starting: %EXPERIMENT_NAME%
    python telegram_notifier.py --status started --task-name "%EXPERIMENT_NAME%" --details "Command: !FULL_COMMAND!"
)

REM Run the command
echo.
echo ================================================
echo Running: %EXPERIMENT_NAME%
echo ================================================
!FULL_COMMAND!

REM Capture exit code
set "EXIT_CODE=!errorlevel!"

REM Send completion notification
if "!NOTIFY_ENABLED!"=="1" (
    if !EXIT_CODE! equ 0 (
        echo [NOTIFY] Success: %EXPERIMENT_NAME%
        python telegram_notifier.py --status success --task-name "%EXPERIMENT_NAME%" --details "Training completed successfully"
    ) else (
        echo [NOTIFY] Failed: %EXPERIMENT_NAME%
        python telegram_notifier.py --status failed --task-name "%EXPERIMENT_NAME%" --details "Training failed with exit code: !EXIT_CODE!"
    )
)

REM Clean up flag file on normal completion or failure
if exist "!FLAG_FILE!" del "!FLAG_FILE!" 2>nul

REM Exit with the original exit code
exit /b !EXIT_CODE!

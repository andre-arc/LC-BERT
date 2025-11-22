@echo off
REM Wrapper script to run training tasks with Telegram notifications
REM Usage: run_with_notification.bat <script_to_run> [args...]
REM Example: run_with_notification.bat run_task_modified.bat 0 3 32

setlocal enabledelayedexpansion

REM Check if a script was provided
if "%~1"=="" (
    echo Error: No script specified
    echo Usage: run_with_notification.bat ^<script_to_run^> [args...]
    echo Example: run_with_notification.bat run_task_modified.bat 0 3 32
    exit /b 1
)

REM Store the script name and all arguments
set "SCRIPT_NAME=%~1"
set "SCRIPT_ARGS="
shift

:parse_args
if "%~1"=="" goto done_parsing
set "SCRIPT_ARGS=!SCRIPT_ARGS! %~1"
shift
goto parse_args

:done_parsing

REM Check if Telegram credentials are set
if "%TELEGRAM_BOT_TOKEN%"=="" (
    echo Warning: TELEGRAM_BOT_TOKEN environment variable not set
    echo Notifications will be skipped
    set "NOTIFY_ENABLED=0"
) else if "%TELEGRAM_CHAT_ID%"=="" (
    echo Warning: TELEGRAM_CHAT_ID environment variable not set
    echo Notifications will be skipped
    set "NOTIFY_ENABLED=0"
) else (
    set "NOTIFY_ENABLED=1"
)

REM Extract experiment name from arguments if present
set "EXPERIMENT_NAME=unknown"
echo !SCRIPT_ARGS! | findstr /C:"--experiment_name" >nul
if !errorlevel! equ 0 (
    for /f "tokens=2 delims= " %%a in ('echo !SCRIPT_ARGS! ^| findstr /C:"--experiment_name"') do (
        set "EXPERIMENT_NAME=%%a"
    )
)

REM Setup flag file path (using temp directory and sanitized task name)
set "FLAG_DIR=%TEMP%\lc-bert-notify"
if not exist "!FLAG_DIR!" mkdir "!FLAG_DIR!"
set "SANITIZED_NAME=%SCRIPT_NAME: =_%"
set "SANITIZED_NAME=!SANITIZED_NAME::=-!"
set "SANITIZED_NAME=!SANITIZED_NAME:.bat=!"
set "FLAG_FILE=!FLAG_DIR!\!SANITIZED_NAME!.running"

REM Check for orphaned flag file from previous interrupted run
if exist "!FLAG_FILE!" (
    if "!NOTIFY_ENABLED!"=="1" (
        echo Found orphaned task from previous run, sending interrupted notification...
        python telegram_notifier.py --status interrupted --task-name "%SCRIPT_NAME%" --experiment-name "%EXPERIMENT_NAME%" --details "Previous run was interrupted or terminated unexpectedly"
    )
    del "!FLAG_FILE!" 2>nul
)

REM Create flag file to track running task
echo %SCRIPT_NAME% > "!FLAG_FILE!"

REM Send start notification
if "!NOTIFY_ENABLED!"=="1" (
    echo Sending start notification...
    python telegram_notifier.py --status started --task-name "%SCRIPT_NAME%" --experiment-name "%EXPERIMENT_NAME%" --details "Arguments: !SCRIPT_ARGS!"
)

REM Run the actual script
echo.
echo ================================================
echo Running: %SCRIPT_NAME% !SCRIPT_ARGS!
echo ================================================
echo.

call "%~dp0%SCRIPT_NAME%" !SCRIPT_ARGS!

REM Capture exit code
set "EXIT_CODE=!errorlevel!"

REM Send completion notification
if "!NOTIFY_ENABLED!"=="1" (
    if !EXIT_CODE! equ 0 (
        echo.
        echo Sending success notification...
        python telegram_notifier.py --status success --task-name "%SCRIPT_NAME%" --experiment-name "%EXPERIMENT_NAME%" --details "Task completed successfully"
    ) else (
        echo.
        echo Sending failure notification...
        python telegram_notifier.py --status failed --task-name "%SCRIPT_NAME%" --experiment-name "%EXPERIMENT_NAME%" --details "Task failed with exit code: !EXIT_CODE!"
    )
)

REM Clean up flag file on normal completion or failure
if exist "!FLAG_FILE!" del "!FLAG_FILE!" 2>nul

REM Exit with the original exit code
exit /b !EXIT_CODE!

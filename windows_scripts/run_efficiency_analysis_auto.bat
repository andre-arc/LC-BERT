@echo off
REM Automated Efficiency Analysis Runner
REM Reads configuration from efficiency_config.txt and runs enabled experiments
REM Usage: run_efficiency_analysis_auto.bat GPU_ID EARLY_STOP BATCH_SIZE

setlocal enabledelayedexpansion

REM Get the directory where this script is located
set SCRIPT_DIR=%~dp0
REM Navigate to project root (parent of windows_scripts)
cd /d "%SCRIPT_DIR%.."

REM Parse command line arguments
set GPU_ID=%1
set EARLY_STOP=%2
set BATCH_SIZE=%3

REM Validate arguments
if "%GPU_ID%"=="" (
    echo Error: GPU_ID is required
    echo Usage: run_efficiency_analysis_auto.bat GPU_ID EARLY_STOP BATCH_SIZE
    echo Example: run_efficiency_analysis_auto.bat 0 3 32
    exit /b 1
)

if "%EARLY_STOP%"=="" (
    echo Error: EARLY_STOP is required
    echo Usage: run_efficiency_analysis_auto.bat GPU_ID EARLY_STOP BATCH_SIZE
    exit /b 1
)

if "%BATCH_SIZE%"=="" (
    echo Error: BATCH_SIZE is required
    echo Usage: run_efficiency_analysis_auto.bat GPU_ID EARLY_STOP BATCH_SIZE
    exit /b 1
)

REM Set environment variable
set CUDA_VISIBLE_DEVICES=%GPU_ID%

REM Configuration file path
set CONFIG_FILE=efficiency_config.txt

REM Check if config file exists
if not exist "%CONFIG_FILE%" (
    echo Error: Configuration file %CONFIG_FILE% not found
    exit /b 1
)

echo ========================================
echo Automated Efficiency Analysis
echo ========================================
echo GPU ID: %GPU_ID%
echo Early Stop: %EARLY_STOP%
echo Batch Size: %BATCH_SIZE%
echo Config File: %CONFIG_FILE%
echo ========================================
echo.

REM Subset percentages to test
set PERCENTAGES=10 20 30 40 50 60 70 80 90 100

REM Read config file and run enabled experiments
for /f "usebackq tokens=1-6 delims=| eol=#" %%a in ("%CONFIG_FILE%") do (
    REM Skip empty lines and process enabled experiments
    if "%%a"=="1" (
        set "MODEL_NAME=%%b"
        set "DATASET=%%c"
        set "EXPERIMENT=%%d"
        set "LR=%%e"
        set "SEED=%%f"

        echo ========================================
        echo Running: !EXPERIMENT!
        echo Dataset: !DATASET!
        echo Model: !MODEL_NAME!
        echo ========================================
        echo.

        REM Run for each percentage
        for %%p in (%PERCENTAGES%) do (
            echo [%%p%%] Running efficiency analysis...

            python efficient_analysis.py ^
                --n_epochs 5 ^
                --train_batch_size %BATCH_SIZE% ^
                --model_name !MODEL_NAME! ^
                --step_size 1 ^
                --gamma 0.9 ^
                --seed !SEED! ^
                --experiment_name !EXPERIMENT! ^
                --lr !LR! ^
                --eps 1e-8 ^
                --early_stop %EARLY_STOP% ^
                --dataset !DATASET! ^
                --lower ^
                --num_layers 2 ^
                --subset_percentage %%p ^
                --force

            if errorlevel 1 (
                echo Error running experiment for %%p%% subset
            ) else (
                echo Successfully completed %%p%% subset
            )
        )

        echo Completed: !EXPERIMENT!
    )
)

echo.
echo ========================================
echo All experiments completed!
echo ========================================
echo.
echo Running aggregation script...
python aggregate_efficiency_results.py --verbose

echo.
echo Analysis complete! Check efficiency_analysis/ directory for results.

endlocal

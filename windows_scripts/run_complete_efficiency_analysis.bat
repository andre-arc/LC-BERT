@echo off
REM Complete Efficiency Analysis Pipeline
REM Runs experiments, aggregates results, and generates visualizations
REM
REM Usage: run_complete_efficiency_analysis.bat GPU_ID EARLY_STOP BATCH_SIZE [--skip-viz]
REM Example: run_complete_efficiency_analysis.bat 0 3 32

setlocal

REM Parse arguments
set GPU_ID=%1
set EARLY_STOP=%2
set BATCH_SIZE=%3
set SKIP_VIZ=%4

REM Validate required arguments
if "%GPU_ID%"=="" (
    echo Error: GPU_ID is required
    echo.
    echo Usage: run_complete_efficiency_analysis.bat GPU_ID EARLY_STOP BATCH_SIZE [--skip-viz]
    echo Example: run_complete_efficiency_analysis.bat 0 3 32
    echo.
    exit /b 1
)

if "%EARLY_STOP%"=="" (
    echo Error: EARLY_STOP is required
    echo.
    echo Usage: run_complete_efficiency_analysis.bat GPU_ID EARLY_STOP BATCH_SIZE [--skip-viz]
    echo Example: run_complete_efficiency_analysis.bat 0 3 32
    echo.
    exit /b 1
)

if "%BATCH_SIZE%"=="" (
    echo Error: BATCH_SIZE is required
    echo.
    echo Usage: run_complete_efficiency_analysis.bat GPU_ID EARLY_STOP BATCH_SIZE [--skip-viz]
    echo Example: run_complete_efficiency_analysis.bat 0 3 32
    echo.
    exit /b 1
)

echo ========================================
echo COMPLETE EFFICIENCY ANALYSIS PIPELINE
echo ========================================
echo.
echo Configuration:
echo   GPU ID: %GPU_ID%
echo   Early Stop: %EARLY_STOP%
echo   Batch Size: %BATCH_SIZE%
echo   Skip Visualization: %SKIP_VIZ%
echo.
echo ========================================
echo.

REM Step 1: Run efficiency analysis
echo [STEP 1/3] Running efficiency experiments...
echo.
call windows_scripts\run_efficiency_analysis_auto.bat %GPU_ID% %EARLY_STOP% %BATCH_SIZE%

if errorlevel 1 (
    echo.
    echo ERROR: Efficiency analysis failed!
    exit /b 1
)

echo.
echo [STEP 1/3] Complete!
echo.

REM Step 2: Aggregate results (already done by auto script, but run again for completeness)
echo [STEP 2/3] Aggregating results...
echo.
python aggregate_efficiency_results.py --verbose

if errorlevel 1 (
    echo.
    echo ERROR: Aggregation failed!
    exit /b 1
)

echo.
echo [STEP 2/3] Complete!
echo.

REM Step 3: Generate visualizations (unless --skip-viz)
if "%SKIP_VIZ%"=="--skip-viz" (
    echo [STEP 3/3] Skipping visualization (--skip-viz flag set)
    goto :complete
)

echo [STEP 3/3] Generating visualizations...
echo.
python visualize_efficiency.py

if errorlevel 1 (
    echo.
    echo WARNING: Visualization failed (but experiments completed successfully)
    echo You can try running: python visualize_efficiency.py
    goto :complete
)

echo.
echo [STEP 3/3] Complete!
echo.

:complete
echo ========================================
echo PIPELINE COMPLETE!
echo ========================================
echo.
echo Results available in:
echo   - efficiency_analysis\all_efficiency_results_*.csv (aggregated data)
echo   - efficiency_analysis\summary_*.csv (summaries)

if not "%SKIP_VIZ%"=="--skip-viz" (
    echo   - efficiency_analysis\plots\ (visualizations^)
)

echo   - save\{dataset}\{experiment}\ (individual results)
echo.
echo To regenerate visualizations: python visualize_efficiency.py
echo To view specific results: explore efficiency_analysis\ directory
echo.

endlocal

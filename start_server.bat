@echo off
:: 设置控制台编码为 UTF-8，防止中文乱码
chcp 65001 >nul
title 漆面缺陷智能检测系统 - 主机控制台 (Host)

echo ========================================================
echo       Car Paint Defect Detection System - HOST
echo ========================================================
echo.

:: 1. 检查 uv 是否安装
uv --version >nul 2>&1
if %errorlevel% neq 0 (
    echo [Error] 未检测到 uv 工具，请先安装 uv！
    pause
    exit /b
)

:: 2. 获取当前脚本所在目录作为根目录
set "PROJECT_ROOT=%~dp0"
cd /d "%PROJECT_ROOT%"
echo [System] 项目根目录: %PROJECT_ROOT%

:: 3. 激活环境并启动
:: 使用 uv run 直接运行，它会自动处理 .venv 环境激活，非常方便
echo [System]正在通过 uv 启动可视化控制台...
echo.

:: 这里的 -- 后面跟参数，比如你可以传参给 python 脚本
uv run framework/core/run_visualizer.py %*

:: 4. 退出处理
if %errorlevel% neq 0 (
    echo.
    echo [Error] 程序异常退出 (Exit Code: %errorlevel%)
    pause
) else (
    echo.
    echo [System] 程序已正常关闭。
    timeout /t 3 >nul
)
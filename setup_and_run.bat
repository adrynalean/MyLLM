@echo off
echo ================================
echo  MyLLM - Setup and Run API
echo ================================

echo Installing dependencies...
pip install fastapi "uvicorn[standard]" tiktoken huggingface_hub pydantic torch

echo.
echo Downloading model from HuggingFace (~197MB)...
python -c "from huggingface_hub import hf_hub_download; hf_hub_download(repo_id='Fluoron/MyLLM', filename='model-ow_best.pkl', local_dir='.')"

echo.
echo Starting API server at http://localhost:8000 ...
echo Press Ctrl+C to stop.
echo.
cd api
python -m uvicorn app:app --port 8000

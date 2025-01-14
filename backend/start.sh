#!/bin/bash
cd /home/ubuntu/kryptos/backend
source /home/ubuntu/kryptos/venv/bin/activate
export PYTHONPATH="/home/ubuntu/kryptos/backend:/home/ubuntu/kryptos/backend/bot:$PYTHONPATH"
python3 -m uvicorn main:app --host 0.0.0.0 --port 8000

#!/usr/bin/env bash
# Start the app, then open http://10.0.0.21:8501
cd "$(dirname "$0")" && exec .venv/bin/streamlit run streamlit_app.py --server.address 0.0.0.0 --server.port 8501

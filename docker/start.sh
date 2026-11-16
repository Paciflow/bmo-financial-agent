#!/bin/bash
set -e

# Both processes in one container: Streamlit talks to FastAPI over localhost,
# exactly like the local two-terminal setup, just co-located instead of two
# separate ports on the host machine. Simplifies the ECS deployment to one
# task/one service instead of coordinating service discovery between two.
uvicorn bmo_agent.api.main:app --host 0.0.0.0 --port 8000 &
API_PID=$!

# Wait for the backend to answer /health before starting Streamlit, so the
# first page load doesn't race the backend's startup.
echo "Waiting for FastAPI backend to become healthy..."
for i in $(seq 1 30); do
    if python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health')" 2>/dev/null; then
        echo "Backend healthy."
        break
    fi
    sleep 1
done

streamlit run app/streamlit_app.py --server.address 0.0.0.0 --server.port 8501 --server.headless true &
STREAMLIT_PID=$!

# If either process dies, the container should die too (so ECS notices and
# restarts it) rather than limping along with half the app missing.
wait -n "$API_PID" "$STREAMLIT_PID"
exit $?

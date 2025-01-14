module.exports = {
  apps: [{
    name: "kryptos-bot",
    cwd: "/home/ubuntu/kryptos/backend",
    script: "/home/ubuntu/kryptos/venv/bin/uvicorn",
    args: "main:app --host 0.0.0.0 --port 8000",
    interpreter: "none",
    env: {
      PYTHONPATH: "/home/ubuntu/kryptos/backend"
    }
  }]
}

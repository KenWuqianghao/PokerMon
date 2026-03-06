"""Serve the PokerMon FastAPI backend + React frontend on Modal.

Usage:
    modal deploy modal_serve.py        # Deploy as a persistent web endpoint
    modal serve modal_serve.py         # Ephemeral dev server
"""

import modal

app = modal.App("pokermon-serve")

COMMIT = "a7a153f"

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git", "curl")
    .run_commands(
        # Node 20 for frontend build
        "curl -fsSL https://deb.nodesource.com/setup_20.x | bash -",
        "apt-get install -y nodejs",
    )
    .run_commands(
        # CPU-only torch — avoid pulling the 2.5 GB CUDA variant
        "pip install torch --index-url https://download.pytorch.org/whl/cpu"
    )
    .pip_install("numpy", "treys", "pyyaml", "fastapi", "uvicorn[standard]")
    .run_commands(
        f"git clone https://github.com/KenWuqianghao/PokerMon.git /root/PokerMon"
        f" && cd /root/PokerMon && git checkout {COMMIT}",
        "pip install -e /root/PokerMon",
        "cd /root/PokerMon/frontend && npm install && npm run build",
    )
)

# Optional: pull updated checkpoints from the training volume
volume = modal.Volume.from_name("pokermon-checkpoints", create_if_missing=True)


@app.function(
    image=image,
    volumes={"/vol": volume},
    cpu=1,
    memory=2048,
    allow_concurrent_inputs=100,
    timeout=300,
)
@modal.asgi_app()
def serve():
    import shutil
    import sys
    from pathlib import Path

    # If a newer model was trained and saved to the volume, use it
    volume_ckpt = Path("/vol/smoke_test.pt")
    repo_ckpt = Path("/root/PokerMon/checkpoints/nlhe6/smoke_test.pt")
    if volume_ckpt.exists():
        volume.reload()
        repo_ckpt.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(volume_ckpt, repo_ckpt)
        print(f"Loaded checkpoint from volume ({volume_ckpt.stat().st_size // 1024} KB)")

    sys.path.insert(0, "/root/PokerMon")
    from server.app import app as fastapi_app

    return fastapi_app

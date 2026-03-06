"""Train Deep CFR on Modal with persistent checkpoints.

CPU-only: MCCFR traversals do single-sample inference where CUDA kernel
launch overhead makes GPU slower than CPU. The model (4-layer 512-dim MLP)
is small enough that CPU SGD with batch 2048 is fast too.

Resumes from the latest checkpoint on the volume after preemption.

Usage:
    modal run --detach modal_train.py              # Train (detached)
    modal run modal_train.py --download            # Download smoke_test.pt locally
"""

import modal

app = modal.App("pokermon-train")

COMMIT = "3c9c980"

image = (
    modal.Image.debian_slim(python_version="3.11")
    .run_commands("pip install torch --index-url https://download.pytorch.org/whl/cpu")
    .pip_install("numpy", "treys", "pyyaml", "tensorboard", "tqdm")
    .run_commands(
        "apt-get update && apt-get install -y git",
        f"git clone https://github.com/KenWuqianghao/PokerMon.git /root/PokerMon"
        f" && cd /root/PokerMon && git checkout {COMMIT}",
        "pip install -e /root/PokerMon",
    )
)

volume = modal.Volume.from_name("pokermon-checkpoints", create_if_missing=True)


def _try_load_checkpoint(checkpoints, advantage_nets, strategy_net):
    """Try loading checkpoints newest-first, skipping corrupt files."""
    from pokermon.train.checkpoint import load_checkpoint

    for ckpt_path in reversed(checkpoints):
        try:
            return load_checkpoint(ckpt_path, advantage_nets, strategy_net), ckpt_path
        except Exception as e:
            print(f"Skipping corrupt checkpoint {ckpt_path.name}: {e}")
    return None, None


@app.function(image=image, cpu=8, memory=32768, timeout=24 * 3600, volumes={"/vol": volume})
def train():
    from pathlib import Path

    import torch

    from pokermon.train.config import TrainConfig
    from pokermon.train.trainer import Trainer

    config = TrainConfig(
        num_players=2,
        hidden_dim=512,
        num_layers=4,
        device="cpu",
        checkpoint_dir="/vol/checkpoints/nlhe_hu",
        log_dir="/vol/runs/nlhe_hu",
        num_iterations=500,
        traversals_per_iter=1500,
        advantage_sgd_steps=4000,
        strategy_sgd_steps=4000,
        checkpoint_every=10,
    )

    trainer = Trainer(config)

    # Resume from latest checkpoint if one exists on the volume
    volume.reload()
    ckpt_dir = Path(config.checkpoint_dir)
    existing = sorted(ckpt_dir.glob("checkpoint_*.pt")) if ckpt_dir.exists() else []
    start_iter = 1

    if existing:
        info, loaded_path = _try_load_checkpoint(
            existing, trainer.advantage_nets, trainer.strategy_net
        )
        if info:
            start_iter = info["iteration"] + 1
            print(f"Resumed from {loaded_path.name} (iteration {info['iteration']})")

    if start_iter > config.num_iterations:
        print(f"Training already complete ({start_iter - 1}/{config.num_iterations})")
    else:
        print(f"Starting from iteration {start_iter}/{config.num_iterations}")
        def on_checkpoint(t: int) -> None:
            # Update smoke_test.pt so the server picks up the latest model live
            checkpoints = sorted(Path(config.checkpoint_dir).glob("checkpoint_*.pt"))
            if checkpoints:
                import torch as _torch
                ckpt = _torch.load(checkpoints[-1], map_location="cpu", weights_only=False)
                _torch.save(ckpt, "/vol/smoke_test.pt")
            volume.commit()

        trainer.train(
            start_iteration=start_iter,
            on_checkpoint=on_checkpoint,
        )

    # Save a portable CPU copy for download
    checkpoints = sorted(Path(config.checkpoint_dir).glob("checkpoint_*.pt"))
    if not checkpoints:
        print("WARNING: No checkpoints found after training")
        volume.commit()
        return

    for ckpt_path in reversed(checkpoints):
        try:
            ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
            torch.save(ckpt, "/vol/smoke_test.pt")
            print(f"Saved /vol/smoke_test.pt from {ckpt_path.name}")
            break
        except Exception as e:
            print(f"Skipping corrupt checkpoint {ckpt_path.name}: {e}")

    volume.commit()


@app.function(image=image, volumes={"/vol": volume})
def fetch_checkpoint():
    from pathlib import Path

    volume.reload()
    src = Path("/vol/smoke_test.pt")
    if not src.exists():
        raise FileNotFoundError("No smoke_test.pt on volume — run training first")
    with open(src, "rb") as f:
        return f.read()


@app.function(image=image, volumes={"/vol": volume})
def upload_checkpoint(data: bytes, name: str):
    """Upload a local checkpoint to the volume (e.g. to seed a new account)."""
    from pathlib import Path

    dest = Path(f"/vol/checkpoints/nlhe_hu/{name}")
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(data)
    volume.commit()
    print(f"Uploaded {name} ({len(data)} bytes) to volume")


@app.local_entrypoint()
def main(download: bool = False, upload: str = ""):
    if download:
        data = fetch_checkpoint.remote()
        with open("smoke_test.pt", "wb") as f:
            f.write(data)
        print(f"Wrote smoke_test.pt ({len(data)} bytes)")
    elif upload:
        with open(upload, "rb") as f:
            data = f.read()
        name = upload.split("/")[-1]
        upload_checkpoint.remote(data, name)
        print(f"Uploaded {name} to volume")
    else:
        train.remote()

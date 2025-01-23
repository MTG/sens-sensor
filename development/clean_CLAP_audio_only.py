from lib.towers import minify_checkpoint
from pathlib import Path

if __name__ == "__main__":
    minify_checkpoint(ckpt=Path("data/models/630k-fusion-best.pt"))

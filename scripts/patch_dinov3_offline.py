"""Make the DINOv3 backbone load without touching the network.

`Dinov3Backbone` calls `torch.hub.load("facebookresearch/dinov3", ..., source="github")`.
With no explicit ref, torch.hub's `_parse_repo_info` urlopens
`github.com/facebookresearch/dinov3/tree/main/` to discover the default branch
*before* it ever checks its own cache, so on a box that cannot reach GitHub the
load hangs indefinitely even with the cache fully populated.

This rewrites the call to `source="local"` against the cached checkout, which is
deterministic and offline. Only the architecture is needed — the original call
already passes `pretrained=False` and the real weights come from model.ckpt.

Idempotent: safe to run repeatedly.
"""

import argparse
import shutil
from pathlib import Path

MARKER = "# patched for offline hub load"

NEW_CALL = '''        # patched for offline hub load
        import os as _os
        _hub_dir = _os.environ.get(
            "SAM3D_DINOV3_DIR",
            _os.path.expanduser("~/.cache/torch/hub/facebookresearch_dinov3_main"),
        )
        if _os.path.isfile(_os.path.join(_hub_dir, "hubconf.py")):
            self.encoder = torch.hub.load(
                _hub_dir,
                self.name,
                source="local",
                pretrained=False,
                drop_path=self.cfg.MODEL.BACKBONE.DROP_PATH_RATE,
            )
        else:
            self.encoder = torch.hub.load(
                "facebookresearch/dinov3:main",
                self.name,
                source="github",
                pretrained=False,
                skip_validation=True,
                drop_path=self.cfg.MODEL.BACKBONE.DROP_PATH_RATE,
            )'''

OLD_CALL = '''        self.encoder = torch.hub.load(
            "facebookresearch/dinov3",
            self.name,
            source="github",
            pretrained=False,
            drop_path=self.cfg.MODEL.BACKBONE.DROP_PATH_RATE,
        )'''


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default="./sam-3d-body")
    args = ap.parse_args()

    f = Path(args.repo) / "sam_3d_body/models/backbones/dinov3.py"
    src = f.read_text()
    if MARKER in src:
        print(f"[skip] already patched: {f}")
        return
    if OLD_CALL not in src:
        raise SystemExit(f"[fail] expected torch.hub.load block not found in {f}; "
                         f"upstream may have changed — patch by hand")
    shutil.copy2(f, f.with_suffix(".py.orig"))
    f.write_text(src.replace(OLD_CALL, NEW_CALL))
    print(f"[ok] patched {f} (original saved as dinov3.py.orig)")


if __name__ == "__main__":
    main()

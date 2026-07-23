#!/usr/bin/env python3
"""Report whether this Python process can access Apple's MPS device."""

from __future__ import annotations

import platform
import subprocess
import sys
import os


try:
    import certifi
except ImportError:  # pragma: no cover - certifi is installed with the project
    certifi = None

if certifi is not None:
    os.environ.setdefault("SSL_CERT_FILE", certifi.where())

import torch


def main() -> int:
    print(f"python: {sys.executable}")
    print(f"python_version: {sys.version.split()[0]}")
    print(f"platform: {platform.platform()}")
    print(f"machine: {platform.machine()}")
    print(f"torch: {torch.__version__}")
    print(f"mps_built: {torch.backends.mps.is_built()}")
    print(f"mps_available: {torch.backends.mps.is_available()}")
    print(f"mps_device_count: {torch.mps.device_count()}")
    print(
        "macos_14_or_newer: "
        f"{torch.backends.mps.is_macos_or_newer(14, 0)}"
    )
    print(
        "sw_vers:\n"
        + subprocess.run(
            ["sw_vers"], check=True, capture_output=True, text=True
        ).stdout.strip()
    )

    if not torch.backends.mps.is_available():
        print("MPS tensor test: unavailable")
        return 1

    try:
        tensor = torch.ones(1, device="mps")
        print(f"MPS tensor test: passed ({tensor.device})")
    except Exception as exc:  # pragma: no cover - depends on host hardware
        print(f"MPS tensor test: failed ({type(exc).__name__}: {exc})")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

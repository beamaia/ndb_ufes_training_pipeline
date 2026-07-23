#!/usr/bin/env python3
"""Check model-specific MPS forward/backward compatibility."""

from __future__ import annotations

import argparse
import gc
import os
import sys

try:
    import certifi
except ImportError:  # pragma: no cover - certifi is installed with the project
    certifi = None

if certifi is not None:
    os.environ.setdefault("SSL_CERT_FILE", certifi.where())

import torch


MODEL_INPUT_SIZES = {
    "efficientnetb4": 380,
    "efficientnet_b1": 240,
    "vit_small_patch16_384": 384,
}

CORE_MODELS = [
    "resnet50",
    "mobilenetv2",
    "densenet121",
    "vgg16",
    "coat_lite_small",
    "pit_s_distilled_224",
    "vit_small_patch16_384",
]

OPTIONAL_MODELS = [
    "vit_base_patch32_224",
    "uni",
    "virchow",
    "ctranspath",
    "mocov3_vit_small",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--models",
        nargs="+",
        default=CORE_MODELS,
        help="Models to check. Optional foundation models can be supplied explicitly.",
    )
    parser.add_argument("--num-classes", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--skip-backward", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not torch.backends.mps.is_available():
        print("MPS is unavailable before model-specific checks.")
        print("Run scripts/check_mps.py in a normal macOS Terminal session.")
        return 2

    sys.path.insert(0, str(__file__).split("/scripts/")[0])
    from src.models.model_selector import ModelSelector

    device = torch.device("mps")
    if args.batch_size < 1:
        raise ValueError("--batch-size must be positive.")
    results: list[tuple[str, str, str]] = []
    print("model\tstatus\tdetails", flush=True)
    for model_name in args.models:
        model = None
        images = labels = output = loss = cpu_output = None
        try:
            model = ModelSelector(model_name, num_classes=args.num_classes).model
            model.to(device).train()
            size = MODEL_INPUT_SIZES.get(model_name, 224)
            images = torch.randn(args.batch_size, 3, size, size, device=device)
            labels = torch.zeros(args.batch_size, dtype=torch.long, device=device)
            output = model(images)
            if output.shape != (args.batch_size, args.num_classes):
                raise RuntimeError(f"unexpected output shape: {tuple(output.shape)}")
            # Match the real training path, which uses weighted cross-entropy.
            class_weights = torch.ones(args.num_classes, device=device)
            loss = torch.nn.functional.cross_entropy(
                output,
                labels,
                weight=class_weights,
            )
            if not args.skip_backward:
                loss.backward()

            status = "mps forward/backward passed"
            torch.mps.synchronize()
            results.append((model_name, "PASS", status))
        except Exception as exc:  # model-specific failures are part of the report
            results.append((model_name, "FAIL", f"{type(exc).__name__}: {exc}"))
        finally:
            model = images = labels = output = loss = None
            gc.collect()
            if hasattr(torch, "mps"):
                torch.mps.empty_cache()
        print("\t".join(results[-1]), flush=True)
    return 0 if all(status == "PASS" for _, status, _ in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())

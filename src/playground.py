"""A small local site to play with the test patches, how they classify their picture, and crops of the picture.

Four views: the test patches (their label and what the models say); the picture of a patch with its test patches, whose call changes as you switch them on and
off; the picture cut into a grid of tiles, each classified; and a free crop drawn by hand. Wherever a crop is (almost) a labelled patch, its true label is shown;
elsewhere only what the models say. Nothing here is a result of the study: it is a way to see the models at work. ``src.playground_export`` writes the same page
as a static site (the stored predictions and every grid computed beforehand).

The server listens on 127.0.0.1 only and never receives a file name (the page sends a picture id and numbers). The patch and picture views use the stored test
predictions (the mean over the five fold runs of each model, as the study reports); the grid and the free crop run the checkpoint of ONE fold run of each model (the
representative one of ``representative_checkpoints.csv``) live, so a call can differ from the stored one.

    uv run python -m src.playground --results ../ndb_ufes_data_organizer/results/experiments/experiment_2
"""

import argparse
import io
import json
import threading
import uuid
from collections import OrderedDict
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

import numpy as np
import pandas as pd
import torch
from PIL import Image

from src import logger
from src.checkpoint import load_checkpoint
from src.config import resolve_device
from src.occlusion import PATCHES, transform_and_white
from src.panel import CACHE, CLASSES, ORGANIZER, PROBABILITIES
from src.select_checkpoints import MLFLOW
from src.unit_consistency import CLASS_NAMES, summarize_predictions  # from there: results_summary alone has a circular import with src.pipeline, which unit_consistency loads first

MODELS = ["vgg16", "densenet121", "swin_tiny_patch4_window7_224"]
TOGETHER = "all three"  # the mean of the probabilities of the models
LAB_TABLE = ORGANIZER / "results/phase0/dataset_tables/private_lab/patch_table_lab.csv"  # private: the SAB picture of each patch and its box
PAGE = Path(__file__).with_suffix(".html")
LOGO = ORGANIZER / "docs/assets/branding/labcin-mark.png"  # the mark of the wiki, whose colours the page uses
SCREEN = 2048  # the pictures are sent with their longer side reduced to this (the pictures are 2048 pixels wide: not reduced)
SMALLEST = 32  # the smallest crop side, in pixels of the picture
KNOWN_IOU = 0.9  # a crop that overlaps a labelled patch this much (intersection over union) is that patch: the page shows its true label
MAX_WINDOWS = 600  # a scan with more windows than this is refused: raise the step
MAX_BODY = 100_000
KEPT_PICTURES = 4  # decoded pictures kept in memory


def window_positions(length: int, size: int, step: int) -> list[int]:
    """Where a window of ``size`` starts along a side of ``length``, every ``step`` pixels, and the last one flush with the edge so that no part is left out."""
    if size > length:
        raise ValueError(f"A window of {size} pixels does not fit in {length}")
    positions = list(range(0, length - size + 1, step))
    return positions if positions[-1] == length - size else positions + [length - size]


def window_grid(width: int, height: int, size: int, step: int) -> list[tuple[int, int]]:
    """The top-left corner of every window of a scan, row by row."""
    return [(x, y) for y in window_positions(height, size, step) for x in window_positions(width, size, step)]


def picture_call(patches: pd.DataFrame) -> dict:
    """The call for a picture from some of its patches, with the rule of the training (``results_summary.summarize_predictions``): the class with the highest mean
    probability, and the label that most patches have (a tie goes to the more severe class). ``patches`` has ``y_true`` and the three probability columns."""
    row = summarize_predictions(patches.assign(picture="p"), CLASS_NAMES, "picture").iloc[0]
    return {"called": CLASSES[int(row["y_pred"])], "label": CLASSES[int(row["y_true"])], "probabilities": [float(row[c]) for c in PROBABILITIES], "patches": int(row["patches"])}


def stored_probabilities(cache: pd.DataFrame, models: list[str]) -> pd.DataFrame:
    """For each test patch and model, the mean probabilities over the fold runs (columns ``prob_*``), indexed by patch and model."""
    test = cache[cache["stage"] == "test"] if "stage" in cache else cache
    return test[test["model"].isin(models)].groupby(["patch_id", "model"])[PROBABILITIES].mean()


class Classifier:
    """One checkpoint, loaded when it is first asked for, that gives the class probabilities of crops (resized to the model's input, square). It runs on the
    device, and on the cpu if the device lacks an operation."""

    def __init__(self, name: str, path: Path, device: str):
        self.name, self.device = name, device
        self.model, bundle = load_checkpoint(path, "cpu", torch.float32)
        self.model.eval()
        self.transform = transform_and_white(bundle["preprocessing"])[0]
        self.moved = False

    def probabilities(self, crops: list[Image.Image], batch: int = 16) -> np.ndarray:
        out = []
        for start in range(0, len(crops), batch):
            images = torch.stack([self.transform(crop) for crop in crops[start:start + batch]])
            while True:
                try:
                    if not self.moved:
                        self.model.to(self.device)
                        self.moved = True
                    with torch.no_grad():
                        out.append(torch.softmax(self.model(images.to(self.device)), dim=-1).cpu().numpy())
                    break
                except RuntimeError as error:
                    if self.device == "cpu":
                        raise
                    logger.warning(f"{self.name} could not run on {self.device} ({str(error)[:80]}...); it goes on the cpu")
                    self.model, self.device, self.moved = self.model.to("cpu"), "cpu", True
        return np.concatenate(out)


class Playground:
    def __init__(self, results: Path, models: list[str], device: str):
        panel = pd.read_csv(results / "panel_patches.csv", index_col="patch_id", keep_default_na=False)
        lab = pd.read_csv(LAB_TABLE, dtype=str, keep_default_na=False).set_index("patch_id")
        cache_file = CACHE / "ndb_ufes_exp2.csv"
        if not cache_file.exists():
            raise SystemExit(f"{cache_file} is missing: run src.panel once (it needs MLflow) to store the test predictions")
        self.stored = stored_probabilities(pd.read_csv(cache_file, keep_default_na=False), models)
        self.models, self.device, self.lock, self.jobs = models, resolve_device(device), threading.Lock(), {}
        self.classifiers: dict[str, Classifier] = {}
        checkpoints = pd.read_csv(results / "representative_checkpoints.csv").set_index("model")["checkpoint"]
        self.checkpoints = {m: MLFLOW / "var/mlflow/artifacts" / checkpoints[m] for m in models}
        table = panel.join(lab[["sab_picture_path", "x", "y", "x2", "y2", "possible_zoom"]])
        table[["x", "y", "x2", "y2"]] = table[["x", "y", "x2", "y2"]].astype(int)
        table["y_true"] = table["true"].map({name: k for k, name in enumerate(CLASSES)})
        self.table = table
        self.paths = table.groupby("histopath_id")["sab_picture_path"].first().map(lambda p: ORGANIZER / p).to_dict()
        self.sizes = {picture: Image.open(path).size for picture, path in self.paths.items()}
        self.screens: dict[str, bytes] = {}
        self.decoded: OrderedDict[str, Image.Image] = OrderedDict()

    # ---- what the page asks for
    def data(self) -> dict:
        """Everything the views need, without a single file name. The pictures are numbered in the order of their ids."""
        pictures = []
        for picture, group in self.table.groupby("histopath_id"):
            width, height = self.sizes[picture]
            patches = [{"id": pid, "x": int(r["x"]), "y": int(r["y"]), "x2": int(r["x2"]), "y2": int(r["y2"]), "label": r["label"], "zoom": r["possible_zoom"], "panel": r["consensus"],
                        "probabilities": {m: [round(v, 4) for v in self.stored.loc[(pid, m)]] for m in self.models}, "thumb": f"/patch/{pid}.png"} for pid, r in group.iterrows()]
            pictures.append({"id": picture, "index": len(pictures) + 1, "image": f"/picture/{picture}.jpg", "width": width, "height": height, "diagnosis": group["parent"].iloc[0], "patches": patches,
                             "check": self.call(picture, [p["id"] for p in patches])})  # the call of the training with every patch on: the page checks its own arithmetic against it
        return {"classes": CLASSES, "models": self.models, "together": TOGETHER, "pictures": pictures, "device": self.device, "known_iou": KNOWN_IOU, "live": True}

    def call(self, picture: str, patch_ids: list[str]) -> dict:
        """The call for the picture from the patches that are on, for each model and for the models together (the mean of their probabilities)."""
        group = self.table[(self.table["histopath_id"] == picture) & self.table.index.isin(patch_ids)]
        if group.empty:
            return {"empty": True}
        frames = {m: group[["y_true"]].join(self.stored.xs(m, level="model")) for m in self.models}
        frames[TOGETHER] = group[["y_true"]].join(sum(self.stored.xs(m, level="model") for m in self.models) / len(self.models))
        return {name: picture_call(frame) for name, frame in frames.items()}

    def picture(self, picture: str) -> Image.Image:
        if picture not in self.decoded:
            with Image.open(self.paths[picture]) as image:
                self.decoded[picture] = image.convert("RGB")
            while len(self.decoded) > KEPT_PICTURES:
                self.decoded.popitem(last=False)
        self.decoded.move_to_end(picture)
        return self.decoded[picture]

    def box(self, picture: str, x: int, y: int, width: int, height: int) -> tuple[int, int, int, int]:
        """A crop checked against the picture: at least SMALLEST pixels a side, and inside the picture."""
        full_width, full_height = self.sizes[picture]
        if min(width, height) < SMALLEST:
            raise ValueError(f"A crop needs at least {SMALLEST} pixels a side")
        if not (0 <= x and 0 <= y and x + width <= full_width and y + height <= full_height):
            raise ValueError("The crop is outside the picture")
        return x, y, x + width, y + height

    def classifier(self, model: str) -> Classifier:
        if model not in self.classifiers:
            logger.info(f"Loading {model}")
            self.classifiers[model] = Classifier(model, self.checkpoints[model], self.device)
        return self.classifiers[model]

    def classify(self, picture: str, box: tuple[int, int, int, int]) -> dict:
        """What each model says about a crop (resized to its input)."""
        crop = self.picture(picture).crop(box)
        with self.lock:
            return {"probabilities": {m: self.classifier(m).probabilities([crop])[0].tolist() for m in self.models}}

    def tile_probabilities(self, picture: str, corners: list[tuple[int, int]], size: int, report=None) -> np.ndarray:
        """The probabilities of every model for the square tiles at ``corners``, shape (tiles, models, classes); the lock is held one batch at a time, so a crop asked for
        during a long job does not wait for all of it. ``report`` is called with the number of tiles done."""
        image, out = self.picture(picture), []
        for start in range(0, len(corners), 16):
            crops = [image.crop((x, y, x + size, y + size)) for x, y in corners[start:start + 16]]
            with self.lock:
                out.append(np.stack([self.classifier(m).probabilities(crops) for m in self.models], axis=1))
            if report:
                report(start + len(crops))
        return np.concatenate(out)

    def start_scan(self, picture: str, size: int, step: int) -> dict:
        width, height = self.sizes[picture]
        corners = window_grid(width, height, size, step)
        if len(corners) > MAX_WINDOWS:
            raise ValueError(f"{len(corners)} windows: raise the step (the limit is {MAX_WINDOWS})")
        job = {"id": uuid.uuid4().hex[:8], "total": len(corners), "done": 0, "corners": corners, "probabilities": [], "size": size, "finished": False, "error": None}
        self.jobs[job["id"]] = job
        threading.Thread(target=self.scan, args=(job, picture), daemon=True).start()
        return {"job": job["id"], "total": job["total"]}

    def scan(self, job: dict, picture: str) -> None:
        try:
            job["probabilities"] = self.tile_probabilities(picture, job["corners"], job["size"], lambda done: job.update(done=done)).round(4).tolist()
        except Exception as error:  # the page shows it
            job["error"] = str(error)
        job["finished"] = True

    def screen(self, picture: str) -> bytes:
        """The picture as a JPEG with its longer side at most SCREEN pixels, kept in memory."""
        if picture not in self.screens:
            image = self.picture(picture).copy()
            image.thumbnail((SCREEN, SCREEN))
            buffer = io.BytesIO()
            image.save(buffer, "JPEG", quality=88)
            self.screens[picture] = buffer.getvalue()
        return self.screens[picture]


def handler(app: Playground) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        def send(self, code: int, body: bytes, kind: str) -> None:
            self.send_response(code)
            self.send_header("Content-Type", kind)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "private, max-age=3600" if kind.startswith("image/") else "no-store")  # the browser keeps the images for an hour, never on disk for others
            self.end_headers()
            self.wfile.write(body)

        def json(self, value, code: int = 200) -> None:
            self.send(code, json.dumps(value).encode(), "application/json")

        def do_GET(self) -> None:
            url = urlparse(self.path)
            path = url.path
            try:
                if path == "/":
                    return self.send(200, PAGE.read_bytes(), "text/html; charset=utf-8")
                if path in ("/data.json", "/api/data"):  # data.json is where the page looks first, here and in a static copy
                    return self.json(app.data())
                if path == "/logo.png" and LOGO.exists():
                    return self.send(200, LOGO.read_bytes(), "image/png")
                if path.startswith("/picture/") and path.endswith(".jpg") and path[9:-4] in app.paths:
                    return self.send(200, app.screen(path[9:-4]), "image/jpeg")
                if path.startswith("/patch/") and path.endswith(".png") and path[7:-4] in app.table.index:
                    return self.send(200, (PATCHES / f"{path[7:-4]}.png").read_bytes(), "image/png")
                if path.startswith("/api/scan/") and path[10:] in app.jobs:
                    job = app.jobs[path[10:]]
                    return self.json({key: job[key] for key in ("total", "done", "finished", "error", "size")} | ({"corners": job["corners"], "probabilities": job["probabilities"]} if job["finished"] else {}))
            except (ValueError, KeyError) as error:
                return self.json({"error": str(error)}, 400)
            self.json({"error": "not found"}, 404)

        def do_POST(self) -> None:
            length = int(self.headers.get("Content-Length", 0))
            if length > MAX_BODY:
                return self.json({"error": "too large"}, 413)
            body = json.loads(self.rfile.read(length) or b"{}")
            try:
                if body.get("picture") not in app.paths:
                    return self.json({"error": "unknown picture"}, 404)
                if self.path == "/api/classify":
                    return self.json(app.classify(body["picture"], app.box(body["picture"], *(int(body[k]) for k in ("x", "y", "w", "h")))))
                if self.path == "/api/scan":
                    return self.json(app.start_scan(body["picture"], int(body["size"]), int(body["step"])))
            except (ValueError, KeyError) as error:
                return self.json({"error": str(error)}, 400)
            self.json({"error": "not found"}, 404)

        def log_message(self, *args) -> None:  # the requests are not worth a line each
            pass

    return Handler


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--results", type=Path, required=True, help="The folder of the experiment in results/experiments, with panel_patches.csv and representative_checkpoints.csv.")
    parser.add_argument("--models", nargs="+", default=MODELS, help="The models of the page (their checkpoints must be restored, see src.select_checkpoints).")
    parser.add_argument("--port", type=int, default=8870, help="Not 8765: another app of this computer uses it.")
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()
    app = Playground(args.results, args.models, args.device)
    server = ThreadingHTTPServer(("127.0.0.1", args.port), handler(app))
    logger.info(f"{len(app.table)} test patches in {len(app.paths)} pictures, models {', '.join(args.models)} on {app.device}. Open http://127.0.0.1:{args.port}/ (Ctrl-C stops it)")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        logger.info("Stopped")


if __name__ == "__main__":
    main()

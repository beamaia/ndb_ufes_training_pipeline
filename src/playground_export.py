"""A static copy of the playground, to put on a website.

It writes one folder with the page, ``data.json`` (the test patches with the stored predictions, and the grid of every picture at every tile size, classified
beforehand by the three models), the pictures, and a small thumbnail of every patch. The page works from these files alone: the free form, which needs the models,
says so. The pictures are numbered; no file name, case name or path of the dataset is written (the folder is checked for them before the command ends).

    uv run python -m src.playground_export --results ../ndb_ufes_data_organizer/results/experiments/experiment_2 --out ../../oscc-playground/site

With ``--bundle <repository of the site>`` it also copies what the container needs to classify a crop drawn by hand: the code that rebuilds a model from its checkpoint
(``src/checkpoint.py`` and ``src/models``) and the three checkpoints (``models/<model>.pt``, about 650 MB: keep them out of git).

The grids take a while (every size from 128 to 1024 pixels in steps of 64, about 18,000 tiles in all, each by three models). Each picture is saved as soon as it is
done, in ``<out>/.grids``, so a stopped run goes on where it stopped; ``--sizes`` and ``--pictures`` make a short trial.
"""

import argparse
import json
import re
import shutil
import time
from pathlib import Path

from PIL import Image

from src import logger
import pandas as pd

from src.occlusion import PATCHES
from src.playground import LOGO, MODELS, PAGE, Playground, window_grid
from src.select_checkpoints import MLFLOW

SIZES = list(range(128, 1025, 64))  # the tile sizes of the page: its slider
THUMB = 128  # the side of a patch thumbnail in pixels
LEAKS = re.compile(r"/Users/|\bSAB\b|hp_[0-9a-f]{6,}|case_\d|private_lab|\.env\b|[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-")  # what a public folder must not hold


def static_data(data: dict, grids: dict[int, dict[int, list]]) -> dict:
    """The page data of the playground with files next to the page and the grids computed beforehand. Pictures are numbered, and the id of a picture is its number."""
    pictures = []
    for picture in data["pictures"]:
        index = picture["index"]
        patches = [patch | {"thumb": f"thumb/{patch['id']}.jpg"} for patch in picture["patches"]]
        pictures.append(picture | {"id": str(index), "image": f"picture/{index}.jpg", "patches": patches})
    return {key: value for key, value in data.items() if key not in ("device", "pictures")} | {"pictures": pictures, "live": False, "grids": grids}


def leaks(folder: Path) -> list[str]:
    """What in the text files of the folder looks like a file name, a case or a path of the dataset."""
    found = []
    for path in sorted(folder.rglob("*")):
        if path.is_file() and path.suffix in (".json", ".html", ".txt", ".md"):
            found += [f"{path.relative_to(folder)}: {match.group(0)}" for match in LEAKS.finditer(path.read_text())]
    return found


def grid_of(app: Playground, picture: str, sizes: list[int]) -> dict[int, list]:
    """For each tile size, the probabilities of the three models for every tile, flat (tiles of ``len(models) * 3`` numbers), rounded to three decimals."""
    width, height = app.sizes[picture]
    return {size: app.tile_probabilities(picture, window_grid(width, height, size, size), size).reshape(-1, 3 * len(app.models)).round(3).tolist() for size in sizes if size <= min(width, height)}


def bundle(results: Path, models: list[str], repository: Path) -> None:
    """Copy into ``repository`` the code that rebuilds a model from a checkpoint (the same import paths as here) and the checkpoint of each model."""
    source = Path(__file__).parent
    (repository / "src").mkdir(parents=True, exist_ok=True)
    (repository / "src/__init__.py").write_text("")
    shutil.copy(source / "checkpoint.py", repository / "src/checkpoint.py")
    shutil.copytree(source / "models", repository / "src/models", dirs_exist_ok=True, ignore=shutil.ignore_patterns("__pycache__"))
    (repository / "models").mkdir(exist_ok=True)
    checkpoints = pd.read_csv(results / "representative_checkpoints.csv").set_index("model")["checkpoint"]
    for model in models:
        shutil.copy(MLFLOW / "var/mlflow/artifacts" / checkpoints[model], repository / "models" / f"{model}.pt")
        logger.info(f"Copied the checkpoint of {model}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--results", type=Path, required=True, help="The folder of the experiment in results/experiments.")
    parser.add_argument("--out", type=Path, required=True, help="The folder to write (made if it does not exist).")
    parser.add_argument("--models", nargs="+", default=MODELS)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--sizes", type=int, nargs="+", default=SIZES, help="The tile sizes of the grids (a trial can use fewer).")
    parser.add_argument("--pictures", type=int, help="Only the first pictures (a trial).")
    parser.add_argument("--bundle", type=Path, help="The repository of the site: also copy into it the model code and the checkpoints, for the container.")
    args = parser.parse_args()

    if args.bundle:
        bundle(args.results, args.models, args.bundle)
    app = Playground(args.results, args.models, args.device)
    data = app.data()
    if args.pictures:
        data["pictures"] = data["pictures"][:args.pictures]
    ids = {picture["index"]: picture["id"] for picture in data["pictures"]}
    cache = args.out / ".grids"
    for folder in (args.out / "picture", args.out / "thumb", cache):
        folder.mkdir(parents=True, exist_ok=True)

    grids, started = {}, time.time()
    for index, picture in ids.items():
        file = cache / f"{index}_{'-'.join(map(str, args.sizes))}.json"
        if not file.exists():
            file.write_text(json.dumps(grid_of(app, picture, args.sizes), separators=(",", ":")))
            logger.info(f"Picture {index} of {len(ids)} classified ({time.time() - started:.0f} s)")
        grids[index] = {int(size): tiles for size, tiles in json.loads(file.read_text()).items()}
        app.picture(picture).save(args.out / "picture" / f"{index}.jpg", quality=85)
    for picture in data["pictures"]:
        for patch in picture["patches"]:
            with Image.open(PATCHES / f"{patch['id']}.png") as image:
                image.convert("RGB").resize((THUMB, THUMB), Image.Resampling.LANCZOS).save(args.out / "thumb" / f"{patch['id']}.jpg", quality=80)

    (args.out / "data.json").write_text(json.dumps(static_data(data, grids), separators=(",", ":")))
    shutil.copy(PAGE, args.out / "index.html")
    if LOGO.exists():
        shutil.copy(LOGO, args.out / "logo.png")
    found = leaks(args.out)
    if found:
        raise SystemExit("The folder holds what must not be public:\n" + "\n".join(found[:20]))
    size = sum(path.stat().st_size for path in args.out.rglob("*") if path.is_file() and ".grids" not in path.parts) / 2**20
    logger.info(f"Wrote {args.out}: {len(ids)} pictures, {sum(len(p['patches']) for p in data['pictures'])} patches, {size:.0f} MB (without .grids, which is only the cache of this command)")


if __name__ == "__main__":
    main()

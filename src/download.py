import shutil
import ssl
import tempfile
import urllib.request
import zipfile
from pathlib import Path

import certifi

from src import logger


def fetch_patches(url: str, patch_dir: Path, patch_ids: list[str]) -> None:
    """Download the public zip (Mendeley Data) and unpack the images of ``patch_ids`` into ``patch_dir``, flat. The zip holds other zips (the patches are in
    one of them), so every zip inside it is searched too. The file name of an image is its patch id. Everything downloaded is deleted afterwards."""
    wanted = {f"{patch_id}.png" for patch_id in patch_ids}
    patch_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=patch_dir.parent) as scratch:
        outer = Path(scratch) / "download.zip"
        logger.info(f"Downloading {url}")
        certificates = ssl.create_default_context(cafile=certifi.where())  # the Python of a Mac may have no certificates of its own
        request = urllib.request.Request(url, headers={"User-Agent": "ndb-ufes-training-pipeline"})  # Mendeley refuses the default Python agent
        with urllib.request.urlopen(request, context=certificates) as response, open(outer, "wb") as target:
            shutil.copyfileobj(response, target, length=1 << 20)
        logger.info(f"Downloaded {outer.stat().st_size / 2**20:.0f} MiB")

        found: dict[str, list[tuple[Path, str]]] = {}  # image name -> where it is: (zip file, member)
        archives = [outer]
        for archive in archives:  # the zip, then the zips inside it, which are added to the list as they are found
            with zipfile.ZipFile(archive) as zipped:
                for member in zipped.namelist():
                    name = Path(member).name
                    if name in wanted:
                        found.setdefault(name, []).append((archive, member))
                    elif name.endswith(".zip"):
                        inner = Path(scratch) / f"inner_{len(archives)}.zip"
                        with zipped.open(member) as source, open(inner, "wb") as target:
                            shutil.copyfileobj(source, target, length=1 << 20)
                        archives.append(inner)
        absent = sorted(wanted - found.keys())
        repeated = sorted(name for name, where in found.items() if len(where) > 1)
        if absent or repeated:
            raise ValueError(f"The zip does not hold exactly one image for each patch: {len(absent)} absent (for example {absent[:3]}), {len(repeated)} in several places (for example {repeated[:3]}); {len(archives)} zip file(s) were searched")

        in_archive: dict[Path, list[tuple[str, str]]] = {}
        for name, ((archive, member),) in found.items():
            in_archive.setdefault(archive, []).append((name, member))
        for archive, images in in_archive.items():
            with zipfile.ZipFile(archive) as zipped:
                for name, member in images:
                    partial = patch_dir / f"{name}.part"
                    with zipped.open(member) as source, open(partial, "wb") as target:
                        shutil.copyfileobj(source, target)
                    partial.replace(patch_dir / name)  # a file that is not complete never has the final name
    logger.info(f"Unpacked {len(wanted)} patch images into {patch_dir}")

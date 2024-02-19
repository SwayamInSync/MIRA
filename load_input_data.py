import glob
import json
import multiprocessing
import os
import platform
import random
import subprocess
import tempfile
import time
import zipfile
from functools import partial
from typing import Any, Dict, List, Literal, Optional, Union

import fsspec
import GPUtil
import pandas as pd
from loguru import logger

import objaverse.xl as oxl
from objaverse.utils import get_uid_from_str

RED = "\033[91m"
ENDC = "\033[0m"


def log_processed_object(csv_filename: str, *args) -> None:
    args = ",".join([str(arg) for arg in args])
    dirname = os.path.expanduser("logs")
    os.makedirs(dirname, exist_ok=True)
    with open(os.path.join(dirname, csv_filename), "a", encoding="utf-8") as f:
        f.write(f"{time.time()},{args}\n")


def zipdir(path: str, ziph: zipfile.ZipFile) -> None:
    for root, dirs, files in os.walk(path):
        for file in files:
            arcname = os.path.join(os.path.basename(root), file)
            ziph.write(os.path.join(root, file), arcname=arcname)


def handle_new_object(
        local_path: str,
        file_identifier: str,
        sha256: str,
        metadata: Dict[str, Any],
        log_file: str = "handle-new-object.csv",
) -> None:
    log_processed_object(log_file, file_identifier, sha256)


def handle_modified_object(
        local_path: str,
        file_identifier: str,
        new_sha256: str,
        old_sha256: str,
        metadata: Dict[str, Any],
        num_renders: int,
        render_dir: str,
        only_northern_hemisphere: bool,
        gpu_devices: Union[int, List[int]],
        render_timeout: int,
) -> None:
    success = handle_found_object(
        local_path=local_path,
        file_identifier=file_identifier,
        sha256=new_sha256,
        metadata=metadata,
        num_renders=num_renders,
        render_dir=render_dir,
        only_northern_hemisphere=only_northern_hemisphere,
        gpu_devices=gpu_devices,
        render_timeout=render_timeout,
        successful_log_file=None,
        failed_log_file=None,
    )

    if success:
        log_processed_object(
            "handle-modified-object-successful.csv",
            file_identifier,
            old_sha256,
            new_sha256,
        )
    else:
        log_processed_object(
            "handle-modified-object-failed.csv",
            file_identifier,
            old_sha256,
            new_sha256,
        )


def handle_missing_object(
        file_identifier: str,
        sha256: str,
        metadata: Dict[str, Any],
        log_file: str = "handle-missing-object.csv",
) -> None:
    log_processed_object(log_file, file_identifier, sha256)

def save_file(local_path: str,
        file_identifier: str,
        sha256: str,
        metadata: Dict[str, Any],
        num_renders: int,
        render_dir: str,
        only_northern_hemisphere: bool,
        gpu_devices: Union[int, List[int]],
        render_timeout: int,
        successful_log_file: Optional[str] = "handle-found-object-successful.csv",
        failed_log_file: Optional[str] = "handle-found-object-failed.csv",
) -> bool:
    file_name = os.path.basename(file_identifier)
    obj_file_path = os.path.join("data", file_name)
    with open(local_path, "rb") as f:
        file_data = f.read()
    with open(obj_file_path, "wb") as f:
        f.write(file_data)
    return True

def handle_found_object(
        local_path: str,
        file_identifier: str,
        sha256: str,
        metadata: Dict[str, Any],
        num_renders: int,
        render_dir: str,
        only_northern_hemisphere: bool,
        gpu_devices: Union[int, List[int]],
        render_timeout: int,
        successful_log_file: Optional[str] = "handle-found-object-successful.csv",
        failed_log_file: Optional[str] = "handle-found-object-failed.csv",
) -> bool:
    save_uid = get_uid_from_str(file_identifier)
    args = f"--object_path '{local_path}' --num_renders {num_renders}"

    using_gpu: bool = True
    gpu_i = 0
    if isinstance(gpu_devices, int) and gpu_devices > 0:
        num_gpus = gpu_devices
        gpu_i = random.randint(0, num_gpus - 1)
    elif isinstance(gpu_devices, list):
        gpu_i = random.choice(gpu_devices)
    elif isinstance(gpu_devices, int) and gpu_devices == 0:
        using_gpu = False
    else:
        raise ValueError(
            f"gpu_devices must be an int > 0, 0, or a list of ints. Got {gpu_devices}."
        )

    with tempfile.TemporaryDirectory() as temp_dir:
        target_directory = os.path.join(temp_dir, save_uid)
        os.makedirs(target_directory, exist_ok=True)
        args += f" --output_dir {target_directory}"

        if platform.system() == "Linux" and using_gpu:
            args += " --engine BLENDER_EEVEE"
        elif platform.system() == "Darwin" or (
                platform.system() == "Linux" and not using_gpu
        ):
            args += " --engine CYCLES"
        else:
            raise NotImplementedError(f"Platform {platform.system()} is not supported.")

        if only_northern_hemisphere:
            args += " --only_northern_hemisphere"

        command = f"blender --background --python blender_scripts/dataset_rendering.py -- {args}"
        if using_gpu:
            command = f"export DISPLAY=:0.{gpu_i} && {command}"

        result = subprocess.run(
            ["bash", "-c", command],
            timeout=render_timeout,
            check=False,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
        )
        if result.returncode != 0:
            logger.error(result.stderr.decode())
        logger.info(command)
        png_files = glob.glob(os.path.join(target_directory, "*.png"))
        metadata_files = glob.glob(os.path.join(target_directory, "*.json"))
        npy_files = glob.glob(os.path.join(target_directory, "*.npy"))
        logger.info(f"png_files: {len(png_files)} | npy_files: {len(npy_files)} | metadata: {len(metadata_files)}")
        if (
                (len(png_files) != num_renders)
                or (len(npy_files) != num_renders)
                or (len(metadata_files) != 1)
        ):
            logger.error(
                f"Found object {file_identifier} was not rendered successfully!"
            )
            if failed_log_file is not None:
                log_processed_object(
                    failed_log_file,
                    file_identifier,
                    sha256,
                )
            return False

        metadata_path = os.path.join(target_directory, "metadata.json")
        with open(metadata_path, "r", encoding="utf-8") as f:
            metadata_file = json.load(f)
        metadata_file["sha256"] = sha256
        metadata_file["file_identifier"] = file_identifier
        metadata_file["save_uid"] = save_uid
        metadata_file["metadata"] = metadata
        with open(metadata_path, "w", encoding="utf-8") as f:
            json.dump(metadata_file, f, indent=2, sort_keys=True)

        # Keeps the {save_uid} directory structure when unzipped
        with zipfile.ZipFile(
                f"{target_directory}.zip", "w", zipfile.ZIP_DEFLATED
        ) as ziph:
            zipdir(target_directory, ziph)

        fs, path = fsspec.core.url_to_fs(render_dir)

        fs.makedirs(os.path.join(path, "renders"), exist_ok=True)
        fs.put(
            os.path.join(f"{target_directory}.zip"),
            os.path.join(path, "renders", f"{save_uid}.zip"),
        )

        if successful_log_file is not None:
            logger.success(
                f"Found object {file_identifier} rendered successfully!"
            )
            log_processed_object(successful_log_file, file_identifier, sha256)

        return True


def sample_annotations(source, n_samples):
    annotations = oxl.get_annotations(download_dir="data/annotations_data")
    github_annotations = annotations[annotations['source'] == source]
    github_annotations.sample(n_samples).to_csv("sample_df", index=False)
    del annotations
    del github_annotations


def get_example_objects() -> pd.DataFrame:
    return pd.read_csv("3D_data.csv")


def render_objects(
        render_dir: str = "data/input_data/",
        download_dir: Optional[str] = "data/downloaded_data",
        num_renders: int = 32,
        processes: Optional[int] = None,
        save_repo_format: Optional[Literal["zip", "tar", "tar.gz", "files"]] = "zip",
        only_northern_hemisphere: bool = False,
        render_timeout: int = 1000,
        gpu_devices: Optional[Union[int, List[int]]] = None,
) -> None:
    if platform.system() not in ["Linux", "Darwin"]:
        raise NotImplementedError(
            f"Platform {platform.system()} is not supported. Use Linux or MacOS."
        )
    if download_dir is None and save_repo_format is not None:
        raise ValueError(
            f"If {save_repo_format=} is not None, {download_dir=} must be specified."
        )
    if download_dir is not None and save_repo_format is None:
        logger.warning(
            f"GitHub repos will not save. While {download_dir=} is specified, {save_repo_format=} None."
        )

    parsed_gpu_devices: Union[int, List[int]] = 0
    if gpu_devices is None:
        parsed_gpu_devices = len(GPUtil.getGPUs())
    logger.info(f"Using {parsed_gpu_devices} GPU devices for rendering.")

    if processes is None:
        processes = multiprocessing.cpu_count() * 3

    objects = get_example_objects()
    # assert objects.iloc[0]["fileType"] == "obj", (
    #             RED + "Currently only renders the .obj file, although just only few lines "
    #                   "of replacement needed to make it work for all formats but "
    #                   "developer is lazy" + ENDC)
    objects = objects.copy()
    logger.info(f"Provided {len(objects)} objects to render.")

    fs, path = fsspec.core.url_to_fs(render_dir)
    try:
        zip_files = fs.glob(os.path.join(path, "renders", "*.zip"), refresh=True)
    except TypeError:
        zip_files = fs.glob(os.path.join(path, "renders", "*.zip"))

    saved_ids = set(zip_file.split("/")[-1].split(".")[0] for zip_file in zip_files)
    logger.info(f"Found {len(saved_ids)} objects already rendered.")

    objects["saveUid"] = objects["fileIdentifier"].apply(get_uid_from_str)
    objects = objects[~objects["saveUid"].isin(saved_ids)]
    objects = objects.reset_index(drop=True)
    logger.info(f"Rendering {len(objects)} new objects.")

    objects = objects.sample(frac=1).reset_index(drop=True)
    oxl.download_objects(
        objects=objects,
        processes=processes,
        save_repo_format=save_repo_format,
        download_dir=download_dir,
        handle_found_object=partial(
            handle_found_object,
            render_dir=render_dir,
            num_renders=num_renders,
            only_northern_hemisphere=only_northern_hemisphere,
            gpu_devices=parsed_gpu_devices,
            render_timeout=render_timeout,
        ),
        handle_new_object=handle_new_object,
        handle_modified_object=partial(
            handle_modified_object,
            render_dir=render_dir,
            num_renders=num_renders,
            only_northern_hemisphere=only_northern_hemisphere,
            gpu_devices=parsed_gpu_devices,
            render_timeout=render_timeout,
        ),
        handle_missing_object=handle_missing_object,
    )


if __name__ == "__main__":
    identifier = "https://github.com/mlivesu/LoopyCuts/blob/c36b81154a03e79208f83725b9f4542f30ee4285/test_data/impeller/impeller.obj"
    annotations = oxl.get_annotations(download_dir="data/annotations_data")
    annotations = annotations[annotations['source'] == "github"]
    # annotations = annotations[(annotations['fileType'] == "obj") & (annotations['source'] == "github")][:10]
    annotations.to_csv("3D_data.csv", index=False)
    render_objects()

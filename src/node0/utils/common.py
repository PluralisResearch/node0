# Copyright 2025 Pluralis Research
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import importlib
import io

from functools import partial
from pathlib import Path
from typing import Any

import requests
import torch

from hivemind.utils.logging import get_logger


logger = get_logger(__name__)


def build_cls(class_path: str, init_args: dict, partial_init: bool = False) -> Any:
    """Instantiate class.

    Args:
        class_path (str): full path to class
        init_args (dict): class init arguments
        partial_init (bool, optional): if True, return partial function. Defaults to False.

    Raises:
        Exception: wrong class path/init arguments

    Returns:
        Any: class instance or partial function
    """
    try:
        # Split module and class names
        module_name, class_name = class_path.rsplit(".", maxsplit=1)

        # Import module and get class
        module = importlib.import_module(module_name)
        class_ = getattr(module, class_name)

        # Instantiate class
        if partial_init:
            instance = partial(class_, **init_args)
        else:
            instance = class_(**init_args)
        return instance
    except Exception as e:
        logger.error(f"Can't initialize class {class_path}: {e}", exc_info=logger.getEffectiveLevel() <= 15)
        raise


def infer_expert_params(
    pipeline_stage: str,
    max_experts: int = 1024,
) -> dict:
    """Infer required expert and stage parameters from pipeline_stage.

    Args:
        pipeline_stage str: stage type in the format: head-X, body-X, tail-X (X is int)
        max_experts (int, optional): max number of experts per stage. Defaults to 1024.

    Raises:
        ValueError: wrong stage type

    Returns:
        dict: expert_pattern, expert_class, stage
    """
    try:
        stage, stage_idx = pipeline_stage.split("-")
        stage_idx = int(stage_idx)
        assert stage in ["head", "body", "tail"]
    except Exception as e:
        raise ValueError("Wrong stage type. It should be one of: head-X, body-X, tail-X (X is int).") from e

    stage_idx_for_pattern = "" if stage in ["head", "tail"] else stage_idx
    expert_pattern = f"{stage}{stage_idx_for_pattern}.0.[32:{max_experts}]"
    stage_name = f"{stage}{stage_idx_for_pattern}.0."
    expert_class = f"lm_{stage}"

    params = {
        "expert_pattern": expert_pattern,
        "expert_cls": expert_class,
        "stage": stage_idx,
        "stage_name": stage_name,
    }

    return params


def load_ss_components(ss_url: str):
    """
    Loads rcv and fixed_embeddings from URL location

    Args:
        ss_url (str): URL to the subspace compression file

    Returns:
        Dict of: rcv, fixed_tok_weight
    """
    try:
        response = requests.get(ss_url, timeout=30)
        response.raise_for_status()

        buffer = io.BytesIO(response.content)
        ss_comp_dict = torch.load(buffer, map_location="cpu", weights_only=True)

        if not isinstance(ss_comp_dict, dict):
            raise RuntimeError("Invalid subspace compression payload format")

        required_keys = {"rcv", "fixed_tok_weight"}
        missing_keys = required_keys - set(ss_comp_dict.keys())
        if missing_keys:
            raise RuntimeError(f"Missing keys in subspace compression payload: {sorted(missing_keys)}")
        for key in required_keys:
            if not isinstance(ss_comp_dict[key], torch.Tensor):
                raise RuntimeError(f"Expected torch.Tensor for key '{key}'")

    except requests.RequestException as e:
        raise RuntimeError(f"Failed to download from URL: {e}") from e
    except RuntimeError as e:
        raise RuntimeError(f"Remote loading of subspace compression components failed: {e}") from e

    return ss_comp_dict


def clean_tmp():
    """Remove tmp hivemind socket files."""
    tmp_dir = Path("/tmp")

    if not tmp_dir.exists():
        return

    try:
        matching_files = list(tmp_dir.glob("hivemind*"))

        if len(matching_files) > 0:
            for fpath in matching_files:
                try:
                    fpath.unlink()
                except Exception as e:
                    logger.error(
                        f"Can't remove file {fpath}: {e}. Please remove all hivemind* files from /tmp folder (see README for instructions to stop the server). Exiting run."
                    )
                    exit(1)

            matching_files = list(tmp_dir.glob("hivemind*"))
            if len(matching_files) > 0:
                logger.error(
                    "Old hivemind* files are found in /tmp folder. Please clean the folder (see README for instructions to stop the server). Exiting run."
                )
                exit(1)
            logger.info("/tmp folder was cleaned")
    except Exception as e:
        logger.warning(f"Can't check tmp folder: {e}")
        return

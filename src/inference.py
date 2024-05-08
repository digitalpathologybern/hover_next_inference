import os
import copy
import toml
import requests
from concurrent.futures import ThreadPoolExecutor
import concurrent.futures
from typing import List, Union, Tuple
import torch
import numpy as np
import zarr
import zipfile
from numcodecs import Blosc
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from scipy.special import softmax
from src.multi_head_unet import get_model, load_checkpoint
from src.data_utils import WholeSlideDataset, NpyDataset, ImageDataset
from src.augmentations import color_augmentations
from src.spatial_augmenter import SpatialAugmenter
from src.constants import TTA_AUG_PARAMS, VALID_WEIGHTS

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def inference_main(
    params: dict,
    models,
    augmenter,
    color_aug_fn,
):
    """
    Inference function for a single input file.

    Parameters
    ----------
    params: dict
        Parameter store, defined in initial main
    models: List[torch.nn.Module]
        list of models to run inference with, e.g. multiple folds or a single model in a list
    augmenter: SpatialAugmenter
        Augmentation module for geometric transformations
    color_aug_fn: torch.nn.Sequential
        Color Augmentation module

    Returns
    ----------
    params: dict
        Parameter store, defined in initial main and modified by this function
    z: Union(Tuple[zarr.ZipStore, zarr.ZipStore], None)
        instance and class segmentation results as zarr stores, kept open for further processing. None if inference was skipped.
    """
    # print(repr(params["p"]))
    fn = params["p"].split(os.sep)[-1].split(params["ext"])[0]
    params["output_dir"] = os.path.join(params["output_root"], fn)
    if not os.path.isdir(params["output_dir"]):
        os.makedirs(params["output_dir"])
    params["model_out_p"] = os.path.join(
        params["output_dir"], fn + "_raw_" + str(params["tile_size"])
    )
    prog_path = os.path.join(params["output_dir"], "progress.txt")

    if os.path.exists(os.path.join(params["output_dir"], "pinst_pp.zip")):
        print(
            "inference and postprocessing already completed, delete output or specify different output path to re-run"
        )
        return params, None

    if (
        os.path.exists(params["model_out_p"] + "_inst.zip")
        & (os.path.exists(params["model_out_p"] + "_cls.zip"))
        & (not os.path.exists(prog_path))
    ):
        try:
            z_inst = zarr.open(params["model_out_p"] + "_inst.zip", mode="r")
            z_cls = zarr.open(params["model_out_p"] + "_cls.zip", mode="r")
            print("Inference already completed", z_inst.shape, z_cls.shape)
            return params, (z_inst, z_cls)
        except (KeyError, zipfile.BadZipFile):
            z_inst = None
            z_cls = None
            print(
                "something went wrong with previous output files, rerunning inference"
            )

    z_inst = None
    z_cls = None

    if not torch.cuda.is_available():
        print("trying to run inference on CPU, aborting...")
        print("if this is intended, remove this check")
        raise Exception("No GPU available")

    # create datasets from specified input

    if params["input_type"] == "npy":
        dataset = NpyDataset(
            params["p"],
            params["tile_size"],
            padding_factor=params["overlap"],
            ratio_object_thresh=0.3,
            min_tiss=0.1,
        )
    elif params["input_type"] == "img":
        dataset = ImageDataset(
            params["p"],
            params["tile_size"],
            padding_factor=params["overlap"],
            ratio_object_thresh=0.3,
            min_tiss=0.1,
        )
    else:
        level = 40 if params["pannuke"] else 20
        dataset = WholeSlideDataset(
            params["p"],
            crop_sizes_px=[params["tile_size"]],
            crop_magnifications=[level],
            padding_factor=params["overlap"],
            remove_background=True,
            ratio_object_thresh=0.0001,
        )

    # setup output files to write to, also create dummy file to resume inference if interruped

    z_inst = zarr.open(
        params["model_out_p"] + "_inst.zip",
        mode="w",
        shape=(len(dataset), 3, params["tile_size"], params["tile_size"]),
        chunks=(params["batch_size"], 3, params["tile_size"], params["tile_size"]),
        dtype="f4",
        compressor=Blosc(cname="lz4", clevel=3, shuffle=Blosc.SHUFFLE),
    )
    z_cls = zarr.open(
        params["model_out_p"] + "_cls.zip",
        mode="w",
        shape=(
            len(dataset),
            params["out_channels_cls"],
            params["tile_size"],
            params["tile_size"],
        ),
        chunks=(
            params["batch_size"],
            params["out_channels_cls"],
            params["tile_size"],
            params["tile_size"],
        ),
        dtype="u1",
        compressor=Blosc(cname="lz4", clevel=3, shuffle=Blosc.BITSHUFFLE),
    )
    # creating progress file to restart inference if it was interrupted
    with open(prog_path, "w") as f:
        f.write("0")
    inf_start = 0

    dataloader = DataLoader(
        dataset,
        batch_size=params["batch_size"],
        shuffle=False,
        num_workers=params["inf_workers"],
        pin_memory=True,
    )

    # IO thread to write output in parallel to inference
    def dump_results(res, z_cls, z_inst, prog_path):
        cls_, inst_, zc_ = res
        if cls_ is None:
            return
        cls_ = (softmax(cls_.astype(np.float32), axis=1) * 255).astype(np.uint8)
        z_cls[zc_ : zc_ + cls_.shape[0]] = cls_
        z_inst[zc_ : zc_ + inst_.shape[0]] = inst_.astype(np.float32)
        with open(prog_path, "w") as f:
            f.write(str(zc_))
        return

    # Separate thread for IO
    with ThreadPoolExecutor(max_workers=params["inf_writers"]) as executor:
        futures = []
        # run inference
        zc = inf_start
        for raw, _ in tqdm(dataloader):
            raw = raw.to(device, non_blocking=True).float()
            raw = raw.permute(0, 3, 1, 2)  # BHWC -> BCHW
            with torch.inference_mode():
                ct, inst = batch_pseudolabel_ensemb(
                    raw, models, params["tta"], augmenter, color_aug_fn
                )
                futures.append(
                    executor.submit(
                        dump_results,
                        (ct.cpu().detach().numpy(), inst.cpu().detach().numpy(), zc),
                        z_cls,
                        z_inst,
                        prog_path,
                    )
                )

                zc += params["batch_size"]

        # Block until all data is written
        for _ in concurrent.futures.as_completed(futures):
            pass
    # clean up
    if os.path.exists(prog_path):
        os.remove(prog_path)
    return params, (z_inst, z_cls)


def batch_pseudolabel_ensemb(
    raw: torch.Tensor,
    models: List[torch.nn.Module],
    nviews: int,
    aug: SpatialAugmenter,
    color_aug_fn: torch.nn.Sequential,
):
    """
    Run inference step on batch of images with test time augmentations

    Parameters
    ----------

    raw: torch.Tensor
        batch of input images
    models: List[torch.nn.Module]
        list of models to run inference with, e.g. multiple folds or a single model in a list
    nviews: int
        Number of test-time augmentation views to aggregate
    aug: SpatialAugmenter
        Augmentation module for geometric transformations
    color_aug_fn: torch.nn.Sequential
        Color Augmentation module

    Returns
    ----------

    ct: torch.Tensor
        Per pixel class predictions as a tensor of shape (batch_size, n_classes+1, tilesize, tilesize)
    inst: torch.Tensor
        Per pixel 3 class prediction map with boundary, background and foreground classes, shape (batch_size, 3, tilesize, tilesize)
    """
    tmp_3c_view = []
    tmp_ct_view = []
    # ensure that at least one view is run, even when specifying 1 view with many models
    if nviews <= 0:
        out_fast = []
        with torch.inference_mode():
            for mod in models:
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    out_fast.append(mod(raw))
        out_fast = torch.stack(out_fast, axis=0).nanmean(0)
        ct = out_fast[:, 5:].softmax(1)
        inst = out_fast[:, 2:5].softmax(1)
    else:
        for _ in range(nviews):
            aug.interpolation = "bilinear"
            view_aug = aug.forward_transform(raw)
            aug.interpolation = "nearest"
            view_aug = torch.clamp(color_aug_fn(view_aug), 0, 1)
            out_fast = []
            with torch.inference_mode():
                for mod in models:
                    with torch.autocast(device_type="cuda", dtype=torch.float16):
                        out_fast.append(aug.inverse_transform(mod(view_aug)))
            out_fast = torch.stack(out_fast, axis=0).nanmean(0)
            tmp_3c_view.append(out_fast[:, 2:5].softmax(1))
            tmp_ct_view.append(out_fast[:, 5:].softmax(1))
        ct = torch.stack(tmp_ct_view).nanmean(0)
        inst = torch.stack(tmp_3c_view).nanmean(0)
    return ct, inst


def get_inference_setup(params):
    """
    get model/ models and load checkpoint, create augmentation functions and set up parameters for inference
    """
    models = []
    for pth in params["data_dirs"]:
        if not os.path.exists(pth):
            pth = download_weights(os.path.split(pth)[-1])               

        checkpoint_path = f"{pth}/train/best_model"
        mod_params = toml.load(f"{pth}/params.toml")
        params["out_channels_cls"] = mod_params["out_channels_cls"]
        params["inst_channels"] = mod_params["inst_channels"]
        model = get_model(
            enc=mod_params["encoder"],
            out_channels_cls=params["out_channels_cls"],
            out_channels_inst=params["inst_channels"],
        ).to(device)
        model = load_checkpoint(model, checkpoint_path, device)
        model.eval()
        models.append(copy.deepcopy(model))
    # create augmentation functions on device
    augmenter = SpatialAugmenter(TTA_AUG_PARAMS).to(device)
    color_aug_fn = color_augmentations(False, rank=device)

    if mod_params["dataset"] == "pannuke":
        params["pannuke"] = True
    else:
        params["pannuke"] = False
    print(
        "processing input using",
        "pannuke" if params["pannuke"] else "lizard",
        "trained model",
    )

    return params, models, augmenter, color_aug_fn

def download_weights(model_code):
    if model_code in VALID_WEIGHTS:
        url = f"https://zenodo.org/records/10635618/files/{model_code}.zip"
        print("downloading",model_code,"weights to",os.getcwd())
        try:
            response = requests.get(url, stream=True, timeout=15.0)
        except requests.exceptions.Timeout:
            print("Timeout")
        total_size = int(response.headers.get("content-length", 0))
        block_size = 1024  # 1 Kibibyte
        with tqdm(total=total_size, unit="iB", unit_scale=True) as t:
            with open("cache.zip", "wb") as f:
                for data in response.iter_content(block_size):
                    t.update(len(data))
                    f.write(data)
        with zipfile.ZipFile("cache.zip", "r") as zip:
            zip.extractall("")
        os.remove("cache.zip")
        return model_code
    else:
        raise ValueError("Model id not found in valid identifiers, please make select one of", VALID_WEIGHTS)

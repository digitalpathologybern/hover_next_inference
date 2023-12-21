import torch
import os
import numpy as np
import zarr
import threading
import multiprocessing
import copy
from numcodecs import Blosc
from torch.utils.data import DataLoader, Subset
from tqdm.auto import tqdm
from scipy.special import softmax

from src.multi_head_unet import get_model, load_checkpoint
from src.data_utils import WholeSlideDataset, NpyDataset
from src.augmentations import color_augmentations
from src.spatial_augmenter import SpatialAugmenter

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
from typing import List, Union, Tuple


def inference_main(
    args: dict,
) -> Tuple[str, str, Union[Tuple[zarr.ZipStore, zarr.ZipStore], None]]:
    """
        Run inference on wsi of different formats as well as ome.tif and .npy stacked images. For npy the format should be e.g. [100,1024,1024,3]

        Parameters
        ----------
        input_path: str
            Path to image.
        output_dir: str
            Where to store the results. A new folder is created for the specific input supplied
        cp_paths: float
            list of paths to checkpoints of all models (folder structure should to be like this):
                                        --{cp_path} \\
                                                     |train \\
                                                            |best_model
                                        --{cp_path} \\
                                                     |train \\
                                                            |best_model
        tile_size: int
            Tile size for the input and extraction from WSI. Model is trained on 256 but fully convolutional so any 512 1024 etc. should be fine
        batch_size: int
            Batch Size to run inference with, optimize for your GPU / CPU setup
        
        Returns
        -------
        output_dir: str
            result directory with filename as folder name
        model_out_p: str
            inference result filename
        (z_inst, z_cls):
            zarr stores with instance and class predictions
            If inference was already completed, returns None
        pannuke: bool
            whether running pannuke processing
        """

    print(repr(args["p"]))
    _, ext = os.path.splitext(args["p"])
    fn = args["p"].split(os.sep)[-1].split(ext)[0]
    args["output_dir"] = os.path.join(args["o"], fn)
    if not os.path.isdir(args["o"]):
        os.makedirs(args["o"])
    args["model_out_p"] = args["output_dir"] + "/" + fn + "_raw_" + args["ts"]
    prog_path = os.path.join(args["output_dir"], "progress.txt")
    if sum(["pannuke" in x for x in args["data_dirs"]]) > 0:
        print("running pannuke inference at .25mpp")
        args["pannuke"] = True
    else:
        args["pannuke"] = False
    if (not os.path.exists(prog_path)) & (
        os.path.exists(args["model_out_p"] + "_inst.zip")
    ):
        print("inference already completed, skipping")
        return args, None

    if device == "cpu":
        print("running inference on cpu, please verify that this is intended")

    # create datasets from specified input

    channels = 6 if args["pannuke"] else 8

    if np.isin(ext, [".npy", ".npz"]):
        dataset = NpyDataset(
            args["p"],
            args["ts"],
            padding_factor=args["ov"],
            ratio_object_thresh=0.3,
            min_tiss=0.1,
        )
    else:
        level = 40 if args["pannuke"] else 20
        dataset = WholeSlideDataset(
            args["p"],
            crop_sizes_px=[args["ts"]],
            crop_magnifications=[level],
            padding_factor=args["ov"],
            remove_background=True,
            ratio_object_thresh=0.0001,
            global_norm=True,
        )

    # setup output files to write to, also create dummy file to resume inference if interruped
    if not os.path.exists(prog_path):
        z_inst = zarr.open(
            args["model_out_p"] + "_inst.zip",
            mode="w",
            shape=(len(dataset), 3, args["ts"], args["ts"]),
            chunks=(args["bs"], 3, args["ts"], args["ts"]),
            dtype="f4",
            compressor=Blosc(cname="lz4", clevel=3, shuffle=Blosc.BITSHUFFLE),
        )
        z_cls = zarr.open(
            args["model_out_p"] + "_cls.zip",
            mode="w",
            shape=(len(dataset), channels, args["ts"], args["ts"]),
            chunks=(args["bs"], channels, args["ts"], args["ts"]),
            dtype="u1",
            compressor=Blosc(cname="lz4", clevel=3, shuffle=Blosc.BITSHUFFLE),
        )
        # creating progress file to restart inference if it was interrupted
        with open(prog_path, "w") as f:
            f.write("0")
        inf_start = 0
    else:
        z_inst = zarr.open(args["model_out_p"] + "_inst.zip", mode="a")
        z_cls = zarr.open(args["model_out_p"] + "_cls.zip", mode="a")
        inf_start = int(open(prog_path, "r").read())
    if inf_start != 0:
        print("resuming inference from", inf_start)
        dataset = Subset(dataset, range(inf_start, len(dataset)))
    # get model/ models and load checkpoint
    models = []
    for pth in args["data_dirs"]:
        checkpoint_path = f"{pth}/train/best_model"
        with open(f"{pth}/params.toml", "r") as f:
            enc = [
                x.split('= "')[1].rstrip('"\n')
                for x in f.readlines()
                if x.startswith("encoder")
            ][0]

        model = get_model(enc=enc, out_channels_cls=channels, out_channels_inst=5).to(
            device
        )
        model, step = load_checkpoint(model, checkpoint_path, device)
        if enc.startswith(("tu-dm_nfnet", "tu-nfnet")):
            model.train()
        else:
            model.eval()
        models.append(copy.deepcopy(model))

    dataloader = DataLoader(
        dataset, batch_size=args["bs"], shuffle=False, num_workers=4, pin_memory=True
    )
    # parameters for test time augmentations, do not change
    aug_params = {
        "mirror": {"prob_x": 0.5, "prob_y": 0.5, "prob": 0.75},
        "translate": {"max_percent": 0.03, "prob": 0.0},
        "scale": {"min": 0.8, "max": 1.2, "prob": 0.0},
        "zoom": {"min": 0.8, "max": 1.2, "prob": 0.0},
        "rotate": {"rot90": True, "prob": 0.75},
        "shear": {"max_percent": 0.1, "prob": 0.0},
        "elastic": {"alpha": [120, 120], "sigma": 8, "prob": 0.0},
    }
    # create augmentation functions on device
    augmenter = SpatialAugmenter(aug_params).to(device)
    color_aug_fn = color_augmentations(False, rank=device)
    # Setup queue for IO thread
    io_queue = multiprocessing.JoinableQueue()

    # IO thread to write output in parallel to inference
    def writer_thread():
        while True:
            cls_, inst_, zc_ = io_queue.get()
            if cls_ is None:
                break
            cls_ = (softmax(cls_.astype(np.float32), axis=1) * 255).astype(np.uint8)
            z_cls[zc_ : zc_ + cls_.shape[0]] = cls_
            z_inst[zc_ : zc_ + inst_.shape[0]] = inst_.astype(np.float32)
            with open(prog_path, "w") as f:
                f.write(str(zc_))
            io_queue.task_done()
        io_queue.task_done()
        print("inference writer thread done...")
        return

    writer = multiprocessing.Process(target=writer_thread, daemon=True)
    writer.start()
    # run inference
    zc = inf_start
    for raw, _ in tqdm(dataloader):
        raw = raw.to(device, non_blocking=True).float()
        raw = raw.permute(0, 3, 1, 2)  # BHWC -> BCHW
        with torch.inference_mode():
            ct, inst = batch_pseudolabel_ensemb(
                raw, models, args["tta"], augmenter, color_aug_fn
            )
            io_queue.put((ct.cpu().detach().numpy(), inst.cpu().detach().numpy(), zc))
            zc += args["bs"]
    io_queue.put((None,None,None))
    io_queue.join()
    # clean up
    if os.path.exists(prog_path):
        os.remove(prog_path)
    return args, (z_inst, z_cls)
    # return output_dir, model_out_p, (z_inst, z_cls), pannuke


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
    for nv in range(nviews // len(models)):
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

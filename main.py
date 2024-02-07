import os
import argparse
import sys
from timeit import default_timer as timer
from datetime import timedelta
import torch
from glob import glob
from src.inference import inference_main, get_inference_setup
from src.post_process import post_process_main
from src.data_utils import copy_img

torch.backends.cudnn.benchmark = True
print(torch.cuda.device_count(), " cuda devices")


def prepare_input(params):
    print("input specified: ", params["input"])
    if params["input"].endswith(".txt"):
        if os.path.exists(params["input"]):
            with open(params["input"], "r") as f:
                input_list = f.read().splitlines()
        else:
            raise FileNotFoundError("input file not found")
    else:
        input_list = sorted(glob(params["input"].rstrip()))
    return input_list


# modify this to support other image input types (supported by opencv)
def get_input_type(params):
    params["ext"] = os.path.splitext(params["p"])[-1]
    if params["ext"] == ".npy":
        params["input_type"] = "npy"
    elif params["ext"] in [".jpg", ".png", ".jpeg", ".bmp"]:
        params["input_type"] = "img"
    else:
        params["input_type"] = "wsi"
    return params


def main(params: dict):
    """
    Start nuclei segmentation and classification pipeline using specified parameters from argparse
    """

    if params["metric"] not in ["mpq", "f1", "pannuke"]:
        params["metric"] = "f1"
        print("invalid metric, falling back to f1")
    else:
        print("optimizing postprocessing for: ", params["metric"])

    params["root"] = os.path.dirname(__file__)
    params["data_dirs"] = [
        os.path.join(params["root"], c) for c in params["cp"].split(",")
    ]

    print("saving results to:", params["output_root"])
    print("loading model from:", params["data_dirs"])

    # Run per tile inference and store results
    params, models, augmenter, color_aug_fn = get_inference_setup(params)

    input_list = prepare_input(params)
    print("Running inference on", len(input_list), "file(s)")

    for inp in input_list:
        start_time = timer()
        params["p"] = inp.rstrip()
        params = get_input_type(params)
        print("Processing ", params["p"])
        if params["cache"] is not None:
            print("Caching input at:")
            params["p"] = copy_img(params["p"], params["cache"])
            print(params["p"])

        params, z = inference_main(params, models, augmenter, color_aug_fn)
        print(
            "::: finished or skipped inference after",
            timedelta(seconds=timer() - start_time),
        )
        process_timer = timer()
        if params["only_inference"]:
            try:
                z[0].store.close()
                z[1].store.close()
            except TypeError:
                # if z is None, z cannot be indexed -> throws a TypeError
                pass
            print("Exiting after inference")
            sys.exit(0)
        # Stitch tiles together and postprocess to get instance segmentation
        if not os.path.exists(os.path.join(params["output_dir"], "pinst_pp.zip")):
            print("running post-processing")

            z_pp = post_process_main(
                params,
                z,
            )
            if not params["keep_raw"]:
                try:
                    os.remove(params["model_out_p"] + "_inst.zip")
                    os.remove(params["model_out_p"] + "_cls.zip")
                except FileNotFoundError:
                    pass
        else:
            z_pp = None
        print(
            "::: postprocessing took",
            timedelta(seconds=timer() - process_timer),
            "total elapsed time",
            timedelta(seconds=timer() - start_time),
        )
        if z_pp is not None:
            z_pp.store.close()
    print("done")


if __name__ == "__main__":
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        type=str,
        default=None,
        help="path to wsi, glob pattern or text file containing paths",
        required=True,
    )
    parser.add_argument(
        "--output_root", type=str, default=None, help="output directory", required=True
    )
    parser.add_argument(
        "--cp",
        type=str,
        default=None,
        help="comma separated list of checkpoint folders to consider",
    )
    parser.add_argument(
        "--only_inference",
        action="store_true",
        help="split inference to gpu and cpu node/ only run inference",
    )
    parser.add_argument(
        "--metric", type=str, default="f1", help="metric to optimize for pp"
    )
    parser.add_argument("--batch_size", type=int, default=64, help="batch size")
    parser.add_argument(
        "--tta",
        type=int,
        default=4,
        help="test time augmentations, number of views (4= results from 4 different augmentations are averaged for each sample)",
    )
    parser.add_argument(
        "--save_polygon",
        action="store_true",
        help="save output as polygons to load in qupath",
    )
    parser.add_argument(
        "--tile_size",
        type=int,
        default=256,
        help="tile size, models are trained on 256x256",
    )
    parser.add_argument(
        "--overlap",
        type=float,
        default=0.96875,
        help="overlap between tiles, for conic, 0.96875 is best, for pannuke use 0.9375 for better results",
    )
    parser.add_argument(
        "--inf_workers",
        type=int,
        default=4,
        help="number of workers for inference dataloader, maximally set this to number of cores",
    )
    parser.add_argument(
        "--inf_writers",
        type=int,
        default=2,
        help="number of writers for inference dataloader, default 2 should be sufficient"
        + ", \ tune based on core availability and delay between final inference step and inference finalization",
    )
    parser.add_argument(
        "--pp_tiling",
        type=int,
        default=8,
        help="tiling factor for post processing, number of tiles per dimension, 8 = 64 tiles",
    )
    parser.add_argument(
        "--pp_overlap",
        type=int,
        default=256,
        help="overlap for postprocessing tiles, put to around tile_size",
    )
    parser.add_argument(
        "--pp_workers",
        type=int,
        default=16,
        help="number of workers for postprocessing, maximally set this to number of cores",
    )
    parser.add_argument(
        "--keep_raw",
        action="store_true",
        help="keep raw predictions (can be large files for particularly for pannuke)",
    )
    parser.add_argument("--cache", type=str, default=None, help="cache path")
    params = vars(parser.parse_args())
    main(params)

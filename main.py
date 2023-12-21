import torch
from src.inference import inference_main
from src.post_process import post_process_main

torch.backends.cudnn.benchmark = True
print(torch.cuda.device_count(), " cuda devices")
import os
import argparse
import sys
from timeit import default_timer as timer
from datetime import timedelta


def main(args: dict):
    """
    Start nuclei segmentation and classification pipeline using specified parameters from argparse
    """

    if args["m"] not in ["mpq", "f1", "hd", "r2"]:
        args["m"] = "f1"
        print("invalid metric, falling back to f1")
    else:
        print("optimizing postprocessing for: ", args["m"])

    args["p"] = args["p"].rstrip()
    args["root"] = os.path.dirname(__file__)
    args["o"] = os.path.join(args["root"], args["o"])
    args["data_dirs"] = [os.path.join(args["root"], c) for c in args["cp"].split(",")]

    print("input path: ", args["p"])
    print("saving results to:", args["o"])
    print("loading model from:", args["data_dirs"])

    # Run per tile inference and store results
    args, z = inference_main(args)
    print(
        "::: finished or skipped inference after",
        timedelta(seconds=timer() - start_time),
    )
    process_timer = timer()
    # for faster processing, only run inference on a GPU node and
    # do post-processing on basic CPU nodes
    if args["slurm"]:
        print("submitting slurm job for postprocessing")
        try:
            z[0].store.close()
            z[1].store.close()
        except:
            pass
        sys.exit(0)
    # Stitch tiles together and postprocess to get instance segmentation
    if not os.path.exists(os.path.join(args["output_dir"], "pinst_pp.zip")):
        print("running post-processing")
        print("downsample: ", args["ds"])
        print("save polygons", args["save_polygon"])

        z_pp = post_process_main(
            args,
            z,
        )
    else:
        z_pp = None
    print(
        "::: postprocessing took",
        timedelta(seconds=timer() - process_timer),
        "total elapsed time",
        timedelta(seconds=timer() - start_time),
    )
    try:
        z_pp.store.close()
    except:
        pass
    print("done")


if __name__ == "__main__":
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)
    start_time = timer()

    parser = argparse.ArgumentParser()
    parser.add_argument("-p", type=str, default=None, help="path to wsi", required=True)
    parser.add_argument(
        "-o", type=str, default=None, help="relative output dir", required=True
    )
    parser.add_argument(
        "-ca", type=str, default=None, help="path to CA segmentation model outputs"
    )
    parser.add_argument(
        "-ds",
        type=int,
        default=0,
        help="downsample results, speed up depends on number of cores available for pp. NPY is never downsampled",
    )
    parser.add_argument(
        "--cp",
        type=str,
        default=None,
        help="comma separated list of checkpoint folders to consider",
    )
    parser.add_argument(
        "--slurm", action="store_true", help="split inference to gpu and cpu node"
    )
    parser.add_argument("-m", type=str, default="f1", help="metric to optimize for pp")
    parser.add_argument("-bs", type=int, default=64, help="batch size")
    parser.add_argument(
        "-tta",
        type=int,
        default=8,
        help="test time augmentations, number of views (8= results from 8 different augmentations are averaged for each sample)",
    )
    parser.add_argument(
        "--save_polygon",
        action="store_true",
        help="save output as polygons to load in qupath",
    )
    parser.add_argument(
        "-ts",
        type=int,
        default=256,
        help="tile size, models are trained on 256x256",
    )
    parser.add_argument(
        "-ov",
        type=float,
        default=0.96875,
        help="overlap between tiles, for conic, 0.96875 is best, for pannuke use 0.875 for better results",
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
        help="overlap for postprocessing tiles, put to at least tile_size",
    )
    parser.add_argument(
        "--pp_workers",
        type=int,
        default=16,
        help="number of workers for postprocessing, maximally set this to number of cores",
    )
    args = vars(parser.parse_args())
    main(args)

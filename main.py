import os
import argparse
import sys
from timeit import default_timer as timer
from datetime import timedelta
import torch
from src.inference import inference_main
from src.post_process import post_process_main

torch.backends.cudnn.benchmark = True
print(torch.cuda.device_count(), " cuda devices")


def main(params: dict):
    """
    Start nuclei segmentation and classification pipeline using specified parameters from argparse
    """

    if params["m"] not in ["mpq", "f1", "hd", "r2"]:
        params["m"] = "f1"
        print("invalid metric, falling back to f1")
    else:
        print("optimizing postprocessing for: ", params["m"])

    params["p"] = params["p"].rstrip()
    params["root"] = os.path.dirname(__file__)
    params["data_dirs"] = [
        os.path.join(params["root"], c) for c in params["cp"].split(",")
    ]

    print("input path: ", params["p"])
    print("saving results to:", params["o"])
    print("loading model from:", params["data_dirs"])

    if sum(["pannuke" in x for x in params["data_dirs"]]) > 0:
        print("running pannuke inference at .25mpp")
        params["pannuke"] = True
    else:
        params["pannuke"] = False
    print(
        "processing input using",
        "pannuke" if params["pannuke"] else "lizard",
        "trained model",
    )

    start_time = timer()
    # Run per tile inference and store results
    params, z = inference_main(params)
    print(
        "::: finished or skipped inference after",
        timedelta(seconds=timer() - start_time),
    )
    process_timer = timer()
    # for faster processing, only run inference on a GPU node and
    # do post-processing on basic CPU nodes
    if params["slurm"]:
        try:
            z[0].store.close()
            z[1].store.close()
        except TypeError:
            # if z is None, z cannot be indexed -> throws a TypeError
            pass
        print("submitting slurm job for postprocessing")
        sys.exit(0)
    # Stitch tiles together and postprocess to get instance segmentation
    if not os.path.exists(os.path.join(params["output_dir"], "pinst_pp.zip")):
        print("running post-processing")
        print("save polygons", params["save_polygon"])

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
    parser.add_argument("-p", type=str, default=None, help="path to wsi", required=True)
    parser.add_argument(
        "-o", type=str, default=None, help="output directory", required=True
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
        default=4,
        help="test time augmentations, number of views (4= results from 4 different augmentations are averaged for each sample)",
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
    parser.add_argument(
        "--keep_raw",
        action="store_true",
        help="keep raw predictions (can be large files for particularly for pannuke)",
    )
    params = vars(parser.parse_args())
    main(params)

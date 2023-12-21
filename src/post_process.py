from src.post_process_utils import (
    work,
    writer_thread,
    get_seed_fg,
    get_shapes,
    get_tile_coords,
)
from src.viz_utils import create_tsvs, create_polygon_output
from src.data_utils import NpyDataset
from typing import List, Tuple
import zarr
from numcodecs import Blosc
import multiprocessing
from concurrent.futures import ProcessPoolExecutor
import concurrent.futures

import numpy as np
import json
import os
from typing import Union


def post_process_main(
    args: dict,
    z: Union[Tuple[zarr.ZipStore, zarr.ZipStore], None] = None,
):
    """
    Post processing function for inference results. Computes stitched output maps and refines prediction results and produces instance and class maps

    Parameters
    ----------

    args: dict
        Parameter store, defined in initial main

    Returns
    ----------
    z_pp: zarr.ZipStore
        instance segmentation results as zarr store, kept open for further processing

    """
    # get best parameters for respective evaluation metric
    best_fg_thresh_cl, best_seed_thresh_cl = get_seed_fg(
        args["root"], args["data_dirs"], False, args["m"]
    )

    out_img_shape, out_cls_shape, ds_coord, ccrop, npy = get_shapes(
        args["p"], len(best_fg_thresh_cl), args["pannuke"], args["ov"], args["ts"]
    )
    args["ccrop"] = ccrop
    args["npy"] = npy
    args["out_img_shape"] = out_img_shape
    args["out_cls_shape"] = out_cls_shape

    tile_crds = get_tile_coords(
        out_img_shape, args["pp_tiling"], pad_size=args["pp_overlap"], npy=npy
    )
    if npy:
        wsis = NpyDataset(
            args["p"],
            args["ts"],
            padding_factor=args["ov"],
            ratio_object_thresh=0.5,
            min_tiss=0.1,
        ).store
        pinst_out = zarr.zeros(
            shape=(out_img_shape[0], *out_img_shape[-2:]),
            dtype="i4",
            compressor=Blosc(cname="lz4", clevel=3, shuffle=Blosc.SHUFFLE),
        )

    else:
        wsis = args["p"]
        pinst_out = zarr.zeros(
            shape=(
                out_img_shape[-1] // (2 ** args["ds"]),
                out_img_shape[-2] // (2 ** args["ds"]),
            ),
            dtype="i4",
            compressor=Blosc(cname="lz4", clevel=3, shuffle=Blosc.SHUFFLE),
        )

    res = multiprocessing.JoinableQueue()
    # numcodecs.blosc.use_threads = True
    executor = ProcessPoolExecutor(max_workers=args["pp_workers"])
    writer = executor.submit(writer_thread, pinst_out, res, args)
    tile_processors = [
        executor.submit(work, tcrd, res, ds_coord, wsis, z, args) for tcrd in tile_crds
    ]
    for future in concurrent.futures.as_completed(tile_processors):
        pass
    res.join()
    res.put((-1, None, None, None, True))
    pinst_out, pcls_out = writer.result()
    res.join()
    res.close()
    executor.shutdown(wait=False)

    print("saving final output")
    if args["output_dir"] is not None:
        zarr.save(os.path.join(args["output_dir"], "pinst_pp.zip"), pinst_out)
    z_pp = pinst_out

    print("storing class dictionary...")
    with open(os.path.join(args["output_dir"], "class_inst.json"), "w") as fp:
        json.dump(pcls_out, fp)

    if not npy:
        print("saving qupath coordinates...")
        create_tsvs(pcls_out, args["output_dir"])
        if args["save_polygon"]:
            create_polygon_output(pinst_out, pcls_out, args["output_dir"], args["ds"])

    return z_pp

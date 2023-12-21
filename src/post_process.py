from src.post_process_utils import (
    work,
    writer_thread,
    get_pp_params,
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
    params: dict,
    z: Union[Tuple[zarr.ZipStore, zarr.ZipStore], None] = None,
):
    """
    Post processing function for inference results. Computes stitched output maps and refines prediction results and produces instance and class maps

    Parameters
    ----------

    params: dict
        Parameter store, defined in initial main

    Returns
    ----------
    z_pp: zarr.ZipStore
        instance segmentation results as zarr store, kept open for further processing

    """
    # get best parameters for respective evaluation metric

    params = get_pp_params(params, True)
    params, ds_coord = get_shapes(params, len(params["best_fg_thresh_cl"]))

    tile_crds = get_tile_coords(
        params["out_img_shape"],
        params["pp_tiling"],
        pad_size=params["pp_overlap"],
        npy=params["npy"],
    )
    if params["npy"]:
        wsis = NpyDataset(
            params["p"],
            params["ts"],
            padding_factor=params["ov"],
            ratio_object_thresh=0.5,
            min_tiss=0.1,
        ).store
        pinst_out = zarr.zeros(
            shape=(params["out_img_shape"][0], *params["out_img_shape"][-2:]),
            dtype="i4",
            compressor=Blosc(cname="lz4", clevel=3, shuffle=Blosc.SHUFFLE),
        )

    else:
        wsis = params["p"]
        pinst_out = zarr.zeros(
            shape=(
                params["out_img_shape"][-1] // (2 ** params["ds"]),
                params["out_img_shape"][-2] // (2 ** params["ds"]),
            ),
            dtype="i4",
            compressor=Blosc(cname="lz4", clevel=3, shuffle=Blosc.SHUFFLE),
        )

    res = multiprocessing.JoinableQueue()
    # numcodecs.blosc.use_threads = True
    executor = ProcessPoolExecutor(max_workers=params["pp_workers"])
    writer = executor.submit(writer_thread, pinst_out, res, params)
    tile_processors = [
        executor.submit(work, tcrd, res, ds_coord, wsis, z, params)
        for tcrd in tile_crds
    ]
    for _ in concurrent.futures.as_completed(tile_processors):
        pass
    res.join()
    res.put((-1, None, None, None, True))
    pinst_out, pcls_out = writer.result()
    res.join()
    res.close()
    executor.shutdown(wait=False)

    print("saving final output")
    if params["output_dir"] is not None:
        zarr.save(os.path.join(params["output_dir"], "pinst_pp.zip"), pinst_out)
    z_pp = pinst_out

    print("storing class dictionary...")
    with open(os.path.join(params["output_dir"], "class_inst.json"), "w") as fp:
        json.dump(pcls_out, fp)

    if not params["npy"]:
        print("saving qupath coordinates...")
        create_tsvs(pcls_out, params)
        if params["save_polygon"]:
            create_polygon_output(
                pinst_out, pcls_out, params["output_dir"], params["ds"]
            )

    return z_pp

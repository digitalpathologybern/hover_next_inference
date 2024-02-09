from src.post_process_utils import (
    work,
    write,
    get_pp_params,
    get_shapes,
    get_tile_coords,
)
from src.viz_utils import create_tsvs, create_polygon_output
from src.data_utils import NpyDataset, ImageDataset
from typing import List, Tuple
import zarr
from numcodecs import Blosc
from concurrent.futures import ProcessPoolExecutor
import concurrent.futures
import json
import os
from typing import Union
from tqdm.auto import tqdm


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
        npy=params["input_type"] != "wsi",
    )
    if params["input_type"] == "wsi":
        pinst_out = zarr.zeros(
            shape=(
                params["out_img_shape"][-1],
                params["out_img_shape"][-2],
            ),
            dtype="i4",
            compressor=Blosc(cname="lz4", clevel=3, shuffle=Blosc.SHUFFLE),
        )

    else:
        pinst_out = zarr.zeros(
            shape=(params["orig_shape"][0], *params["orig_shape"][-2:]),
            dtype="i4",
            compressor=Blosc(cname="lz4", clevel=3, shuffle=Blosc.SHUFFLE),
        )

    executor = ProcessPoolExecutor(max_workers=params["pp_workers"])
    tile_processors = [
        executor.submit(work, tcrd, ds_coord, z, params) for tcrd in tile_crds
    ]
    pcls_out = {}
    running_max = 0
    for future in tqdm(
        concurrent.futures.as_completed(tile_processors), total=len(tile_processors)
    ):
        pinst_out, pcls_out, running_max = write(
            pinst_out, pcls_out, running_max, future.result(), params
        )
    executor.shutdown(wait=False)


    if params["output_dir"] is not None:
        print("saving final output")
        zarr.save(os.path.join(params["output_dir"], "pinst_pp.zip"), pinst_out)
        print("storing class dictionary...")
        with open(os.path.join(params["output_dir"], "class_inst.json"), "w") as fp:
            json.dump(pcls_out, fp)

        if params["input_type"] == "wsi":
            print("saving geojson coordinates for qupath...")
            create_tsvs(pcls_out, params)
            # TODO this is way to slow for large images
            if params["save_polygon"]:
                create_polygon_output(pinst_out, pcls_out, params["output_dir"], params)
   
    return pinst_out

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
import numpy as np
from typing import Union
from tqdm.auto import tqdm
from src.viz_utils import create_geojson
from src.constants import (
    CLASS_LABELS_LIZARD,
    CLASS_LABELS_PANNUKE,
)


def save_class_probs(pcls_out, prob_id_blocks, prob_mat_blocks, params):
    """Stream per-nucleus class probabilities to class_probs.csv.

    Rows are emitted for exactly the nuclei present in ``pcls_out`` (the
    source of truth), so entries dropped during overlap resolution are
    excluded without needing to remove them from the probability blocks.
    """
    if not prob_id_blocks:
        return
    ids = np.concatenate(prob_id_blocks)
    mat = np.vstack(prob_mat_blocks)
    if len(ids) == 0:
        return
    order = np.argsort(ids, kind="stable")
    sorted_ids = ids[order]
    sorted_mat = mat[order]
    # keep only the last occurrence of each id: later tiles override
    # earlier ones during overlap resolution
    keep = np.ones(len(sorted_ids), dtype=bool)
    keep[:-1] = sorted_ids[1:] != sorted_ids[:-1]
    sorted_ids = sorted_ids[keep]
    sorted_mat = sorted_mat[keep]

    pred_keys = CLASS_LABELS_PANNUKE if params["pannuke"] else CLASS_LABELS_LIZARD
    id_to_name = {v: k for k, v in pred_keys.items()}
    n_classes = sorted_mat.shape[1] if sorted_mat.ndim == 2 else 0
    header = ["id", "class"] + [id_to_name.get(i + 1, str(i + 1)) for i in range(n_classes)]

    out_path = os.path.join(params["output_dir"], "class_probs.csv")
    print("saving class probabilities...")
    with open(out_path, "w") as f:
        f.write(",".join(header) + "\n")
        for k in pcls_out:
            nucleus_id = int(k)
            pos = np.searchsorted(sorted_ids, nucleus_id)
            if pos >= len(sorted_ids) or sorted_ids[pos] != nucleus_id:
                continue
            cls = int(pcls_out[k][0])
            name = id_to_name.get(cls, str(cls))
            probs_str = ",".join(f"{p:.6f}" for p in sorted_mat[pos])
            f.write(f"{nucleus_id},{name},{probs_str}\n")


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
    prob_id_blocks = []
    prob_mat_blocks = []
    running_max = 0
    class_labels = []
    res_poly = []
    for future in tqdm(
        concurrent.futures.as_completed(tile_processors), total=len(tile_processors)
    ):
        result = write(
            pinst_out, pcls_out, running_max, future.result(), params, class_labels, res_poly
        )
        if len(result) == 6:
            pinst_out, pcls_out, running_max, class_labels, res_poly, probs_out = result
            prob_id_blocks.append(probs_out[0])
            prob_mat_blocks.append(probs_out[1])
        else:
            pinst_out, pcls_out, running_max, class_labels, res_poly = result
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
                pred_keys = CLASS_LABELS_PANNUKE if params["pannuke"] else CLASS_LABELS_LIZARD
                create_geojson(
                    res_poly,
                    class_labels,
                    dict((v, k) for k, v in pred_keys.items()),
                    params,
                )

        if params.get("save_probabilities", False):
            save_class_probs(pcls_out, prob_id_blocks, prob_mat_blocks, params)

    return pinst_out

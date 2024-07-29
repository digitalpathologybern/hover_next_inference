import os
import numpy as np
import geojson
import openslide
import cv2
from skimage.measure import regionprops
from src.post_process_utils import get_openslide_info
from src.constants import (
    CLASS_LABELS_LIZARD,
    CLASS_LABELS_PANNUKE,
    COLORS_LIZARD,
    CONIC_MPP,
    PANNUKE_MPP,
)


def create_geojson(polygons, classids, lookup, params):
    features = []
    if isinstance(classids[0], (list, tuple)):
        classids = [cid[0] for cid in classids]
    for i, (poly, cid) in enumerate(zip(polygons, classids)):
        poly = np.array(poly)
        poly = poly[:, [1, 0]] * params["ds_factor"]
        poly = poly.tolist()
        # poly.append(poly[0])
        feature = geojson.Feature(
            geometry=geojson.LineString(poly, precision=2),
            properties={
                "Name": f"Nuc {i}",
                "Type": "Polygon",
                "classification": {
                    "name": lookup[cid],
                    "color": COLORS_LIZARD[cid - 1],
                },
            },
        )
        features.append(feature)
    feature_collection = geojson.FeatureCollection(features)
    with open(params["output_dir"] + "/poly.geojson", "w") as outfile:
        geojson.dump(feature_collection, outfile)


def create_tsvs(pcls_out, params):
    pred_keys = CLASS_LABELS_PANNUKE if params["pannuke"] else CLASS_LABELS_LIZARD

    coord_array = np.array([[i[0], *i[1]] for i in pcls_out.values()])
    classes = list(pred_keys.keys())
    colors = ["-256", "-65536"]
    i = 0
    for pt in classes:
        file = os.path.join(params["output_dir"], "pred_" + pt + ".tsv")
        textfile = open(file, "w")

        textfile.write("x" + "\t" + "y" + "\t" + "name" + "\t" + "color" + "\n")
        textfile.writelines(
            [
                str(element[2] * params["ds_factor"])
                + "\t"
                + str(element[1] * params["ds_factor"])
                + "\t"
                + pt
                + "\t"
                + colors[0]
                + "\n"
                for element in coord_array[coord_array[:, 0] == pred_keys[pt]]
            ]
        )

        textfile.close()
        i += 1


def cont(x):
    _, im, bb = x
    im = np.pad(im.astype(np.uint8), 1, mode="constant", constant_values=0)

    # initial contour finding
    cont = cv2.findContours(
        im,
        mode=cv2.RETR_EXTERNAL,
        method=cv2.CHAIN_APPROX_TC89_KCOS,
    )[0][0].reshape(-1, 2)[:, [1, 0]]
    # since opencv does not do "pixel" contours, we artificially do this for single pixel detections (if they exist)
    if cont.shape[0] <= 1:
        im = cv2.resize(im, None, fx=2.0, fy=2.0)
        cont = (
            cv2.findContours(
                im,
                mode=cv2.RETR_EXTERNAL,
                method=cv2.CHAIN_APPROX_TC89_KCOS,
            )[0][0].reshape(-1, 2)[:, [1, 0]]
            / 2.0
        )
    cont = (cont + bb[0:2] - 1).tolist()
    # close polygon:
    cont.append(cont[0])
    return cont


def create_polygon_output(pinst, pcls_out, params):
    # polygon output is slow and unwieldy, TODO
    pred_keys = CLASS_LABELS_PANNUKE if params["pannuke"] else CLASS_LABELS_LIZARD
    # whole slide regionprops could be avoided to speed up this process...
    print("getting all detections...")
    props = [(p.label, p.image, p.bbox) for p in regionprops(np.asarray(pinst))]
    class_labels = [pcls_out[str(p[0])] for p in props]
    print("generating contours...")
    res_poly = [cont(i) for i in props]
    print("creating output...")
    create_geojson(
        res_poly,
        class_labels,
        dict((v, k) for k, v in pred_keys.items()),
        params,
    )

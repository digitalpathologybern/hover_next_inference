import numpy as np
from skimage.measure import find_contours, regionprops
import geojson
from multiprocessing import Pool
from functools import partial
from tqdm.auto import tqdm


#### HARD CODED CLASSES AND COLORS! ####
colors_ = [
    [0, 255, 0],  # neu
    [255, 0, 0],  # epi
    [0, 0, 255],  # lym
    [0, 128, 0],  # pla
    [0, 255, 255],  # eos
    [255, 179, 102],  # con
    [255, 0, 255],  # mitosis
]

pred_keys = {
    "neutrophil": 1,
    "epithelial-cell": 2,
    "lymphocyte": 3,
    "plasma-cell": 4,
    "eosinophil": 5,
    "connective-tissue-cell": 6,
    "mitosis": 7,
}


def create_geojson(polygons, classids, lookup, result_dir):
    features = []
    for i, (poly, cid) in enumerate(zip(polygons, classids)):
        poly = np.array(poly)
        poly = poly[:, [1, 0]]
        poly = poly.tolist()
        # poly.append(poly[0])
        feature = geojson.Feature(
            geometry=geojson.LineString(poly),
            properties={
                "Name": f"Nuc {i}",
                "Type": "Polygon",
                "classification": {
                    "name": lookup[cid],
                    "color": colors_[cid - 1],
                },
            },
        )
        features.append(feature)
    feature_collection = geojson.FeatureCollection(features)
    geojson_str = geojson.dumps(feature_collection, indent=2)
    with open(result_dir + "/poly.geojson", "w") as f:
        f.write(geojson_str)


def create_tsvs(pcls_out, result_dir):
    coord_array = np.array([[i[0], *i[1]] for i in pcls_out.values()])
    classes = list(pred_keys.keys())
    colors = ["-256", "-65536"]
    i = 0
    for pt in classes:
        file = result_dir + "/pred_" + pt + ".tsv"
        textfile = open(file, "w")

        textfile.write("x" + "\t" + "y" + "\t" + "name" + "\t" + "color" + "\n")
        textfile.writelines(
            [
                str(element[2])
                + "\t"
                + str(element[1])
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


def cont(x, downsample=0):
    lab, im, bb = x
    return (
        lab,
        (
            np.around(
                (
                    np.array(find_contours(np.pad(im, 1, mode="constant"), 0.5)[0])
                    + bb[0:2]
                )
            )
        ).tolist(),
    )


def create_polygon_output(pinst, pcls_out, result_dir, downsample=0):
    # polygon output is slow and unwieldy, TODO
    props = [(p.label, p.image, p.bbox) for p in tqdm(regionprops(np.asarray(pinst)))]
    class_labels = [pcls_out[str(p[0])] for p in props]
    with Pool(4) as pool:
        res_poly = [
            y[1]
            for y in sorted(
                pool.map(partial(cont, downsample=downsample), props),
                key=lambda x: x[0],
            )
        ]
    create_geojson(
        res_poly,
        class_labels,
        dict((v, k) for k, v in pred_keys.items()),
        result_dir,
    )

### Size thresholds for nuclei (in pixels), pannuke is less conservative
# These have been optimized for the conic challenge, but can be changed
# to get more small nuclei (e.g. by setting all min_threshs to 0)
MIN_THRESHS_LIZARD = [30, 30, 20, 20, 30, 30, 15]
MAX_THRESHS_LIZARD = [5000, 5000, 5000, 5000, 5000, 5000, 5000]
MIN_THRESHS_PANNUKE = [10, 10, 10, 10, 10]
MAX_THRESHS_PANNUKE = [20000, 20000, 20000, 3000, 10000]

# Maximal size of holes to remove from a nucleus
MAX_HOLE_SIZE = 128

# Colors for geojson output
COLORS_LIZARD = [
    [0, 255, 0],  # neu
    [255, 0, 0],  # epi
    [0, 0, 255],  # lym
    [0, 128, 0],  # pla
    [0, 255, 255],  # eos
    [255, 179, 102],  # con
    [255, 0, 255],  # mitosis
]

COLORS_PANNUKE = [
    [255, 0, 0],  # neo
    [0, 127, 255],  # inf
    [255, 179, 102],  # con
    [0, 0, 0],  # dead
    [0, 255, 0],  # epi
]

# text labels for lizard
CLASS_LABELS_LIZARD = {
    "neutrophil": 1,
    "epithelial-cell": 2,
    "lymphocyte": 3,
    "plasma-cell": 4,
    "eosinophil": 5,
    "connective-tissue-cell": 6,
    "mitosis": 7,
}

# text labels for pannuke
CLASS_LABELS_PANNUKE = {
    "neoplastic": 1,
    "inflammatory": 2,
    "connective": 3,
    "dead": 4,
    "epithelial": 5,
}

# magnifiation and resolutions for WSI dataloader
LUT_MAGNIFICATION_X = [10, 20, 40, 80]
LUT_MAGNIFICATION_MPP = [0.97, 0.485, 0.2425, 0.124]

CONIC_MPP = 0.5
PANNUKE_MPP = 0.25

# parameters for test time augmentations, do not change
TTA_AUG_PARAMS = {
    "mirror": {"prob_x": 0.5, "prob_y": 0.5, "prob": 0.75},
    "translate": {"max_percent": 0.03, "prob": 0.0},
    "scale": {"min": 0.8, "max": 1.2, "prob": 0.0},
    "zoom": {"min": 0.8, "max": 1.2, "prob": 0.0},
    "rotate": {"rot90": True, "prob": 0.75},
    "shear": {"max_percent": 0.1, "prob": 0.0},
    "elastic": {"alpha": [120, 120], "sigma": 8, "prob": 0.0},
}

# current valid pre-trained weights to be automatically downloaded and used in HoVer-NeXt
VALID_WEIGHTS = [
    "lizard_convnextv2_large",
    "lizard_convnextv2_base",
    "lizard_convnextv2_tiny",
    "pannuke_convnextv2_tiny_1",
    "pannuke_convnextv2_tiny_2",
    "pannuke_convnextv2_tiny_3",
]
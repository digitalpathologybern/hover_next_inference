### Size thresholds for nuclei (in pixels), pannuke is less conservative
# These have been optimized for the conic challenge, but can be changed
# to get more small nuclei (e.g. by setting all min_threshs to 0)
MIN_THRESHS_LIZARD = [30, 30, 20, 20, 30, 30, 15]  # stick to best conic
MAX_THRESHS_LIZARD = [5000, 5000, 5000, 5000, 5000, 5000, 5000]  # stick to best conic

MIN_THRESHS_PANNUKE = [10, 10, 10, 10, 10]
MAX_THRESHS_PANNUKE = [20000, 20000, 20000, 3000, 10000]

MAX_HOLE_SIZE = 128

COLORS_LIZARD = [
    [0, 255, 0],  # neu
    [255, 0, 0],  # epi
    [0, 0, 255],  # lym
    [0, 128, 0],  # pla
    [0, 255, 255],  # eos
    [255, 179, 102],  # con
    [255, 0, 255],  # mitosis
]

CLASS_LABELS_LIZARD = {
    "neutrophil": 1,
    "epithelial-cell": 2,
    "lymphocyte": 3,
    "plasma-cell": 4,
    "eosinophil": 5,
    "connective-tissue-cell": 6,
    "mitosis": 7,
}

CLASS_LABELS_PANNUKE = {
    "neoplastic": 1,
    "inflammatory": 2,
    "connective": 3,
    "dead": 4,
    "epithelial": 5,
}

LUT_MAGNIFICATION_X = [20, 40, 80]
LUT_MAGNIFICATION_MPP = [0.485, 0.2425, 0.124]

CONIC_MPP = 0.5
PANNUKE_MPP = 0.25

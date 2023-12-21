### Size thresholds for nuclei (in pixels), pannuke is less conservative
# These have been optimized for the conic challenge, but can be changed
# to get more small nuclei (e.g. by setting all min_threshs to 0)
MIN_THRESHS_LIZARD = [30, 30, 20, 20, 30, 30, 15]  # stick to best conic
MAX_THRESHS_LIZARD = [5000, 5000, 5000, 5000, 5000, 5000, 5000]  # stick to best conic

MIN_THRESHS_PANNUKE = [10, 10, 10, 10, 10]
MAX_THRESHS_PANNUKE = [20000, 20000, 20000, 3000, 10000]

MAX_HOLE_SIZE = 128

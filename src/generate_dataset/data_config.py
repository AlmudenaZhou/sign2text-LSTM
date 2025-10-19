import os
import numpy as np


DATA_PATH = os.path.join("./data/video")

ACTIONS = np.array(
    [
        "cafe",
        "chocolate",
        "contento",
        "encantar",
        "hoy",
        "parado",
        "quieres",
        "tomar",
        "yo",
    ]
)

NUM_VIDEOS = 100

VIDEO_FRAME_LENGTH = 40

MAX_LENGTH_SENTENCE = 3

# Number of equal frames to consider a new word in the video
FRAME_CONSISTENCY_NUMBER = 5

FACE_POINTS = (24, 3)
ORIGINAL_FACE_LANDMARK_DIMENSIONS = (468, 3)
LANDMARK_DIMENSIONS = ((33, 4), (21, 3), (21, 3))
FULL_LANDMARK_DIMENSION = 1662

POSE_IDX_LEFT_ARM = [11, 13, 15, 17, 19, 21]
POSE_IDX_RIGHT_ARM = [12, 14, 16, 18, 20, 22]
POSE_IDX_CORE = [11, 12]
POSE_IDX_ARMS_CORE = POSE_IDX_LEFT_ARM + POSE_IDX_RIGHT_ARM + POSE_IDX_CORE

USED_LANDMARK_DIMENSION = (
    FACE_POINTS[0] * FACE_POINTS[1] + len(POSE_IDX_ARMS_CORE) * 4 + 21 * 3 * 2
)

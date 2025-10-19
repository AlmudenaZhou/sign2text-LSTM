import os

import numpy as np

from ..video.live_video import get_train_dataset_video

from .data_config import ACTIONS, DATA_PATH, NUM_VIDEOS


def create_dataset_folders():
    for action in ACTIONS:
        try:
            action_dir_files = os.listdir(os.path.join(DATA_PATH, action))
        except FileNotFoundError:
            action_dir_files = None

        dirmax = 1
        if action_dir_files:
            dirmax = (
                np.max(
                    np.array(os.listdir(os.path.join(DATA_PATH, action))).astype(int)
                )
                + 1
            )
        for sequence in range(dirmax, dirmax + NUM_VIDEOS):
            os.makedirs(os.path.join(DATA_PATH, action, str(sequence)), exist_ok=True)

    return dirmax


def generate_dataset():
    start_folder = create_dataset_folders()
    get_train_dataset_video(camera_index=0, start_folder=start_folder)

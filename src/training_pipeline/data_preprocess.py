import os

import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical

from ..video.mediapipe_landmarks import extract_face_oval_indexes

from ..generate_dataset.data_config import (
    ACTIONS,
    DATA_PATH,
    FULL_LANDMARK_DIMENSION,
    VIDEO_FRAME_LENGTH,
    POSE_IDX_ARMS_CORE,
    FACE_POINTS,
    LANDMARK_DIMENSIONS,
    ORIGINAL_FACE_LANDMARK_DIMENSIONS,
)


def __assign_labels_to_actions():
    return {label: num for num, label in enumerate(ACTIONS)}


def __remove_unused_pose_landmarks(all_landmarks):
    last_face_idx = FACE_POINTS[0] * FACE_POINTS[1]
    last_pose_idx = last_face_idx + (
        LANDMARK_DIMENSIONS[0][0] * LANDMARK_DIMENSIONS[0][1]
    )
    pose_landmarks = all_landmarks[last_face_idx:last_pose_idx]
    pose_landmarks = pose_landmarks.reshape(LANDMARK_DIMENSIONS[0])
    new_pose_landmarks = pose_landmarks[POSE_IDX_ARMS_CORE, :].flatten()
    return np.concatenate(
        [
            all_landmarks[:last_face_idx],
            new_pose_landmarks,
            all_landmarks[last_pose_idx:],
        ]
    )


def __filter_face_landmarks(all_landmarks):
    all_face_dim = (
        ORIGINAL_FACE_LANDMARK_DIMENSIONS[0] * ORIGINAL_FACE_LANDMARK_DIMENSIONS[1]
    )
    face_landmarks = all_landmarks[:all_face_dim].reshape(
        ORIGINAL_FACE_LANDMARK_DIMENSIONS
    )
    face_indexes = extract_face_oval_indexes(n_points=FACE_POINTS[0])
    face_vec = face_landmarks[face_indexes, :].flatten()
    return np.concatenate([face_vec, all_landmarks[all_face_dim:]])


def __get_data():
    label_map = __assign_labels_to_actions()
    sequences, labels = [], []
    all_face_dim = (
        ORIGINAL_FACE_LANDMARK_DIMENSIONS[0] * ORIGINAL_FACE_LANDMARK_DIMENSIONS[1]
    )
    diff_face_dim = all_face_dim - FACE_POINTS[0] * FACE_POINTS[1]
    for action in ACTIONS:
        for sequence in np.array(os.listdir(os.path.join(DATA_PATH, action))).astype(
            int
        ):
            window = []
            for frame_num in range(VIDEO_FRAME_LENGTH):
                res = np.load(
                    os.path.join(
                        DATA_PATH, action, str(sequence), "{}.npy".format(frame_num)
                    )
                )
                if res.shape == (FULL_LANDMARK_DIMENSION,):
                    res = __filter_face_landmarks(res)

                if res.shape == (FULL_LANDMARK_DIMENSION - diff_face_dim,):
                    res = __remove_unused_pose_landmarks(res)
                window.append(res)
            sequences.append(window)
            labels.append(label_map[action])

    features = np.array(sequences)
    labels = np.array(labels)
    return features, labels


def get_train_test_data():
    features, labels = __get_data()
    y = to_categorical(labels).astype(int)
    X_train, X_test, y_train, y_test = train_test_split(features, y, test_size=0.05)
    return X_train, X_test, y_train, y_test

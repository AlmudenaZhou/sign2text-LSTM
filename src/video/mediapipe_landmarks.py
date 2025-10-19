from typing import Tuple
import cv2
import mediapipe as mp
import numpy as np
from mediapipe.framework.formats import landmark_pb2

from ..generate_dataset.data_config import (
    FACE_POINTS,
    LANDMARK_DIMENSIONS,
    POSE_IDX_ARMS_CORE,
)


mp_drawing = mp.solutions.drawing_utils
mp_fm = mp.solutions.face_mesh
mp_pose = mp.solutions.pose
mp_hands = mp.solutions.hands


def mediapipe_detection(frame, holistic):
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False
    results = holistic.process(image)
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    return image, results


def _ordered_face_oval_indices():
    edges = list(mp_fm.FACEMESH_FACE_OVAL)
    adj = {}
    for i, j in edges:
        adj.setdefault(i, []).append(j)
        adj.setdefault(j, []).append(i)
    start = next(iter(adj))
    ordered, prev, curr = [start], None, start
    while True:
        n0, n1 = adj[curr]
        nxt = n0 if n0 != prev else n1
        if nxt == start:
            break
        ordered.append(nxt)
        prev, curr = curr, nxt
    return ordered


def extract_face_oval_indexes(n_points: int):
    idx = _ordered_face_oval_indices()
    sampled_idx = idx
    if n_points < len(idx):
        pos = np.linspace(0, len(idx) - 1, n_points)
        sampled_idx = [idx[int(round(p))] for p in pos]
    return sampled_idx


def extract_face_oval(face_landmarks, n_points: int = 24):
    if not face_landmarks:
        return np.zeros(n_points * 3, dtype=np.float32)

    sampled_idx = extract_face_oval_indexes(n_points)

    pts = np.array(
        [
            [
                face_landmarks.landmark[i].x,
                face_landmarks.landmark[i].y,
                face_landmarks.landmark[i].z,
            ]
            for i in sampled_idx
        ],
        dtype=np.float32,
    )
    return pts


def __get_landmarks(results):
    face_landmarks = results.face_landmarks
    pose_landmarks = results.pose_landmarks
    left_hand_landmarks = results.left_hand_landmarks
    right_hand_landmarks = results.right_hand_landmarks
    return (face_landmarks, pose_landmarks, left_hand_landmarks, right_hand_landmarks)


def draw_custom_landmarks(
    image,
    original_landmarks,
    original_connections,
    custom_idx,
    close_loop=False,
    point_spec=None,
    line_spec=None,
):
    landmark_subset = landmark_pb2.NormalizedLandmarkList(
        landmark=[original_landmarks.landmark[i] for i in custom_idx]
    )

    connections = []
    if original_connections:
        keep_set = set(custom_idx)
        index_map = {orig: new for new, orig in enumerate(custom_idx)}
        for a, b in original_connections:
            if a in keep_set and b in keep_set:
                connections.append((index_map[a], index_map[b]))

    if close_loop:
        connections = [(i, (i + 1) % len(custom_idx)) for i in range(len(custom_idx))]

    mp_drawing.draw_landmarks(
        image,
        landmark_subset,
        connections=connections,
        landmark_drawing_spec=point_spec,
        connection_drawing_spec=line_spec,
    )


def draw_landmarks(image, results):
    (face_landmarks, pose_landmarks, left_hand_landmarks, right_hand_landmarks) = (
        __get_landmarks(results)
    )

    if face_landmarks:
        face_sampled_idx = extract_face_oval_indexes(n_points=FACE_POINTS[0])
        point_spec = mp_drawing.DrawingSpec(
            color=(80, 110, 10), thickness=1, circle_radius=1
        )
        line_spec = mp_drawing.DrawingSpec(
            color=(80, 256, 121), thickness=1, circle_radius=1
        )
        draw_custom_landmarks(
            image,
            face_landmarks,
            mp_fm.FACEMESH_FACE_OVAL,
            face_sampled_idx,
            close_loop=True,
            line_spec=line_spec,
            point_spec=point_spec,
        )

    if pose_landmarks:
        point_spec = mp_drawing.DrawingSpec(
            color=(80, 22, 10), thickness=2, circle_radius=4
        )
        line_spec = mp_drawing.DrawingSpec(
            color=(80, 44, 121), thickness=2, circle_radius=2
        )
        draw_custom_landmarks(
            image,
            pose_landmarks,
            mp_pose.POSE_CONNECTIONS,
            POSE_IDX_ARMS_CORE,
            point_spec=point_spec,
            line_spec=line_spec,
        )

    if left_hand_landmarks:
        mp_drawing.draw_landmarks(
            image,
            left_hand_landmarks,
            mp_hands.HAND_CONNECTIONS,
            mp_drawing.DrawingSpec(color=(121, 22, 76), thickness=2, circle_radius=4),
            mp_drawing.DrawingSpec(color=(121, 44, 250), thickness=2, circle_radius=2),
        )

    if right_hand_landmarks:
        mp_drawing.draw_landmarks(
            image,
            right_hand_landmarks,
            mp_hands.HAND_CONNECTIONS,
            mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=4),
            mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2),
        )


def __process_landmarks(landmarks, dimension: Tuple[int, int]):
    if dimension[1] == 3:
        landmarks = (
            np.array([[res.x, res.y, res.z] for res in landmarks.landmark]).flatten()
            if landmarks
            else np.zeros(dimension[0] * dimension[1])
        )
    elif dimension[1] == 4:
        landmarks = (
            np.array(
                [[res.x, res.y, res.z, res.visibility] for res in landmarks.landmark]
            ).flatten()
            if landmarks
            else np.zeros(dimension[0] * dimension[1])
        )
    return landmarks


def extract_keypoints(results):
    face_landmarks, *landmark_results = __get_landmarks(results)

    face_vec = extract_face_oval(face_landmarks, n_points=FACE_POINTS[0]).flatten()

    landmark_keypoints = [
        __process_landmarks(landmarks, landmark_dimensions)
        for landmarks, landmark_dimensions in zip(landmark_results, LANDMARK_DIMENSIONS)
    ]
    landmark_keypoints[0] = (
        landmark_keypoints[0].reshape(33, 4)[POSE_IDX_ARMS_CORE, :].flatten()
    )
    return np.concatenate([face_vec, *landmark_keypoints])

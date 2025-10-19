import os
from contextlib import contextmanager
import cv2
import mediapipe as mp
import numpy as np

from ..generate_dataset.data_config import (
    ACTIONS,
    DATA_PATH,
    NUM_VIDEOS,
    VIDEO_FRAME_LENGTH,
    MAX_LENGTH_SENTENCE,
    FRAME_CONSISTENCY_NUMBER,
)
from .mediapipe_landmarks import draw_landmarks, extract_keypoints, mediapipe_detection


@contextmanager
def video_session(camera_index=0, window_name="camera"):
    cap = cv2.VideoCapture(camera_index)
    window_name = "OpenCV Feed"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
    cv2.resizeWindow(window_name, 1080, 810)

    try:
        with mp.solutions.holistic.Holistic(
            min_detection_confidence=0.5, min_tracking_confidence=0.5
        ) as holistic:
            yield cap, holistic
    finally:
        if cap.isOpened():
            cap.release()
        cv2.destroyAllWindows()


def video_with_landmarks(camera_index: int = 0):
    with video_session(camera_index) as (cap, holistic):
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            image, results = mediapipe_detection(frame, holistic)
            draw_landmarks(image, results)

            cv2.imshow("OpenCV Feed", image)
            if cv2.waitKey(10) & 0xFF == ord("q") or (
                cv2.getWindowProperty("OpenCV Feed", cv2.WND_PROP_VISIBLE) < 1
            ):
                break


def __write_on_screen(image, text, org: cv2.typing.Point, font_scale, color, thickness):
    cv2.putText(
        image,
        text,
        org,
        cv2.FONT_HERSHEY_SIMPLEX,
        font_scale,
        color,
        thickness,
        cv2.LINE_AA,
    )


def get_train_dataset_video(camera_index: int = 0, start_folder: int = 1):
    with video_session(camera_index) as (cap, holistic):
        for action in ACTIONS:
            for sequence in range(start_folder, start_folder + NUM_VIDEOS):
                for frame_num in range(VIDEO_FRAME_LENGTH):
                    _, frame = cap.read()

                    image, results = mediapipe_detection(frame, holistic)
                    draw_landmarks(image, results)
                    if frame_num == 0:
                        __write_on_screen(
                            image, "STARTING COLLECTION", (120, 200), 1, (0, 255, 0), 4
                        )

                    collection_text = "Collecting frames for {} Video Number {}".format(
                        action, sequence
                    )
                    __write_on_screen(
                        image, collection_text, (15, 12), 0.5, (0, 0, 255), 1
                    )

                    keypoints = extract_keypoints(results)
                    npy_path = os.path.join(
                        DATA_PATH, action, str(sequence), str(frame_num)
                    )
                    np.save(npy_path, keypoints)

                    cv2.imshow("OpenCV Feed", image)
                    if cv2.waitKey(10) & 0xFF == ord("q") or (
                        cv2.getWindowProperty("OpenCV Feed", cv2.WND_PROP_VISIBLE) < 1
                    ):
                        break


def video_live_for_inference(model, camera_index: int = 0):
    sequence = []
    sentence = []
    predictions = []

    with video_session(camera_index) as (cap, holistic):
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            image, results = mediapipe_detection(frame, holistic)
            draw_landmarks(image, results)
            keypoints = extract_keypoints(results)
            sequence.append(keypoints)
            sequence = sequence[-VIDEO_FRAME_LENGTH:]

            if len(sequence) == VIDEO_FRAME_LENGTH:
                res = model.predict(np.expand_dims(sequence, axis=0))[0]
                predictions.append(np.argmax(res))
                if np.unique(predictions[-FRAME_CONSISTENCY_NUMBER:]).shape[
                    0
                ] == 1 and (
                    not (len(sentence) > 0 and ACTIONS[np.argmax(res)] == sentence[-1])
                    and ACTIONS[np.argmax(res)] != "parado"
                ):
                    sentence.append(ACTIONS[np.argmax(res)])

            cv2.rectangle(image, (0, 0), (640, 40), (245, 117, 16), -1)
            sentence_text = " ".join(sentence)
            __write_on_screen(image, sentence_text, (3, 30), 1, (255, 255, 255), 2)

            if len(sentence) > MAX_LENGTH_SENTENCE:
                sentence = sentence[-MAX_LENGTH_SENTENCE:]

            cv2.imshow("OpenCV Feed", image)
            if cv2.waitKey(10) & 0xFF == ord("q") or (
                cv2.getWindowProperty("OpenCV Feed", cv2.WND_PROP_VISIBLE) < 1
            ):
                break

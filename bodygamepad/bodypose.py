"""Recognize the body pose of the player."""

import collections
import enum
import math
import os
import pathlib
from typing import List, Final, Mapping, Callable, MutableMapping, Tuple, Any

import cv2
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
from icontract import require


class KeypointLabel(enum.Enum):
    """Map keypoints names to the indices in the network output."""

    NOSE = 0
    LEFT_EYE = 1
    RIGHT_EYE = 2
    LEFT_EAR = 3
    RIGHT_EAR = 4
    LEFT_SHOULDER = 5
    RIGHT_SHOULDER = 6
    LEFT_ELBOW = 7
    RIGHT_ELBOW = 8
    LEFT_WRIST = 9
    RIGHT_WRIST = 10
    LEFT_HIP = 11
    RIGHT_HIP = 12
    LEFT_KNEE = 13
    RIGHT_KNEE = 14
    LEFT_ANKLE = 15
    RIGHT_ANKLE = 16


#: Map indices of the network output to keypoint labels
KEYPOINT_INDEX_TO_LABEL = {literal.value: literal for literal in KeypointLabel}


class Keypoint:
    """Represent a single detection of a keypoint in an image."""

    x: float
    y: float
    confidence: float

    @require(lambda confidence: 0 <= confidence <= 1)
    def __init__(self, x: float, y: float, confidence: float) -> None:
        """
        Initialize with the given values.

        :param x: X-coordinate in the image rescaled to [0, 1] x [0, 1]
        :param y: Y-coordinate in the image rescaled to [0, 1] x [0, 1]
        :param confidence: in the range [0,1] of the keypoint detection
        """
        self.x = x
        self.y = y
        self.confidence = confidence


class Detection:
    """Represent a detection of a person in an image."""

    #: Keypoints of the pose
    keypoints: Final[Mapping[KeypointLabel, Keypoint]]

    #: Score of the person detection.
    #:
    #: .. note::
    #:
    #:     This score is the score of the *person* detection, not of the individual
    #:     joints. For the score of the individual joints,
    #:     see :py:attr:`Keypoint.confidence`

    @require(lambda score: 0 <= score <= 1)
    def __init__(
        self,
        keypoints: Mapping[KeypointLabel, Keypoint],
        score: float,
    ) -> None:
        """Initialize with the given values."""
        self.keypoints = keypoints
        self.score = score


def load_empty_detector() -> Callable[[cv2.Mat], List[Detection]]:
    """
    Create a detector that always returns an empty result.

    This is used for debugging.
    """

    def apply_model(img: cv2.Mat) -> List[Detection]:
        return []

    return apply_model


# noinspection SpellCheckingInspection
def _load_tf_model(path: pathlib.Path) -> Any:
    """
    Load the TF model from disk.

    This function is an adaption of ``tensorflow_hub.load(.)``
    """
    module_path = str(path)

    is_hub_module_v1 = tf.io.gfile.exists(
        hub.native_module.get_module_proto_path(module_path)
    )

    saved_model_path = os.path.join(
        tf.compat.as_bytes(module_path),
        tf.compat.as_bytes(tf.saved_model.SAVED_MODEL_FILENAME_PB),
    )
    saved_model_pbtxt_path = os.path.join(
        tf.compat.as_bytes(module_path),
        tf.compat.as_bytes(tf.saved_model.SAVED_MODEL_FILENAME_PBTXT),
    )
    if not tf.io.gfile.exists(saved_model_path) and not tf.io.gfile.exists(
        saved_model_pbtxt_path
    ):
        raise ValueError(
            f"Trying to load a model of incompatible/unknown type. "
            f"{module_path} contains neither {tf.saved_model.SAVED_MODEL_FILENAME_PB} "
            f"nor {tf.saved_model.SAVED_MODEL_FILENAME_PBTXT}."
        )

    obj = tf.compat.v1.saved_model.load_v2(module_path)
    obj._is_hub_module_v1 = is_hub_module_v1  # pylint: disable=protected-access
    return obj


@require(lambda path: path.exists() and path.is_dir())
def load_detector(path: pathlib.Path) -> Callable[[cv2.Mat], List[Detection]]:
    """
    Load the model and return the function which you can readily use on images.

    :param path: to the model directory
    :return: detector function to be applied on images
    """
    model = _load_tf_model(path)

    movenet = model.signatures["serving_default"]

    # If a detection has a score below this threshold, it will be ignored.
    detection_score_threshold = 0.2

    # If a keypoint has a confidence below this threshold, it will be ignored.
    keypoint_confidence_threshold = 0.2

    def apply_model(img: cv2.Mat) -> List[Detection]:
        # NOTE (mristin, 2023-02-26):
        # Vaguely based on:
        # * https://www.tensorflow.org/hub/tutorials/movenet,
        # * https://www.section.io/engineering-education/multi-person-pose-estimator-with-python/,
        # * https://analyticsindiamag.com/how-to-do-pose-estimation-with-movenet/ and
        # * https://github.com/geaxgx/openvino_movenet_multipose/blob/main/MovenetMPOpenvino.py

        # Both height and width need to be multiple of 32,
        # height to width ratio should resemble the original image, and
        # the larger side should be made to 256 pixels.
        #
        # Example: 720x1280 should be resized to 160x256.

        height, width, _ = img.shape

        input_size = 256

        if height > width:
            new_height = input_size
            # fmt: off
            new_width = int(
                (float(width) * float(new_height) / float(height)) // 32
            ) * 32
            # fmt: on
        else:
            new_width = input_size
            # fmt: off
            new_height = int(
                (float(height) * float(new_width) / float(width)) // 32
            ) * 32
            # fmt: on

        if new_height != height or new_width != width:
            resized = cv2.resize(img, (new_width, new_height))
        else:
            resized = img

        tf_input_img = tf.cast(
            tf.image.resize_with_pad(
                image=tf.expand_dims(resized, axis=0),
                target_height=new_height,
                target_width=new_width,
            ),
            dtype=tf.int32,
        )

        inference = movenet(tf_input_img)
        output_as_tensor = inference["output_0"]
        assert output_as_tensor.shape == (1, 6, 56)

        output = np.squeeze(output_as_tensor)
        assert output.shape == (6, 56)

        detections = []  # type: List[Detection]

        for detection_i in range(6):
            kps = output[detection_i][:51].reshape(17, -1)
            bbox = output[detection_i][51:55].reshape(2, 2)
            score = output[detection_i][55]

            if score < detection_score_threshold:
                continue

            assert kps.shape == (17, 3)
            assert bbox.shape == (2, 2)

            kps_xy = kps[:, [1, 0]]
            kps_confidence = kps[:, 2]

            assert kps_xy.shape == (17, 2)
            assert kps_confidence.shape == (17,)

            keypoints = (
                collections.OrderedDict()
            )  # type: MutableMapping[KeypointLabel, Keypoint]

            for keypoint_i in range(17):
                label = KEYPOINT_INDEX_TO_LABEL[keypoint_i]
                kp_x, kp_y = kps_xy[keypoint_i, :]
                kp_confidence = kps_confidence[keypoint_i]

                if kp_confidence < keypoint_confidence_threshold:
                    continue

                assert label not in keypoints
                keypoints[label] = Keypoint(kp_x, kp_y, kp_confidence)

            detection = Detection(keypoints, score)

            detections.append(detection)

        return detections

    return apply_model


@require(
    lambda hip, knee, ankle: hip[1] > ankle[1] and knee[1] > ankle[1],
    "Coordinate origin in the bottom-left of the image, not in the top-left",
)
def compute_knee_angle(
    hip: Tuple[float, float], knee: Tuple[float, float], ankle: Tuple[float, float]
) -> float:
    """
    Compute the angle between the knee and the other two points.

    Going right means the negative angle:
    >>> round(compute_knee_angle((0, 2), (1, 1), (0, 0)))
    -90

    Squatting means smaller angle when going to the right:
    >>> round(compute_knee_angle((0, 0.5), (1, 0.25), (0, 0)))
    -28

    Going left means the positive angle:
    >>> round(compute_knee_angle((0, 2), (-1, 1), (0, 0)))
    90

    Squatting means smaller angle also to the left:
    >>> round(compute_knee_angle((0, 0.5), (-1, 0.25), (0, 0)))
    28

    Going straight means 180:
    >>> round(compute_knee_angle((0, 2), (0, 1), (0, 0)))
    180

    Some observations regarding the body pose:

    * It seems that going left/right is indicated when the angle goes below 150 degrees.
    * The bent of 120 degrees is already a stretch for the body.
    * 90 degrees is almost impossible.
    """
    # See: https://stackoverflow.com/a/31334882/1600678
    rads = math.atan2(hip[1] - knee[1], hip[0] - knee[0]) - math.atan2(
        ankle[1] - knee[1], ankle[0] - knee[0]
    )

    degrees = rads / math.pi * 180.0
    if degrees > 180:
        # NOTE (mristin, 2023-02-26):
        # We transform the angle so that we can simply compare the sign for
        # the direction and the magnitude for the speed.
        degrees = -(360 - degrees)

    return degrees

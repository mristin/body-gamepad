"""Emulate arrow keys and two buttons with your arms and legs."""

import argparse
import enum
import importlib
import math
import os
import pathlib
import sys
from typing import Optional, Tuple

import cv2
import keyboard
import numpy as np
from icontract import require, ensure

import bodygamepad
from bodygamepad import bodypose
from bodygamepad import common

assert bodygamepad.__doc__ == __doc__

PACKAGE_DIR = (
    pathlib.Path(str(importlib.resources.files(__package__)))  # type: ignore
    if __package__ is not None
    else pathlib.Path(os.path.realpath(__file__)).parent
)


class Direction(enum.Enum):
    """List the possible directions of the gamepad."""

    NOWHERE = 0
    NORTH = 1
    NORTH_EAST = 2
    EAST = 3
    SOUTH_EAST = 4
    SOUTH = 5
    SOUTH_WEST = 6
    WEST = 7
    NORTH_WEST = 8


DIRECTION_FROM_VALUE = {direction.value: direction for direction in Direction}


class Hand(enum.Enum):
    """List hands that steer the directions."""

    LEFT = 0
    RIGHT = 1


class Leg(enum.Enum):
    """List legs that steer the buttons."""

    LEFT = 0
    RIGHT = 1


class Button(enum.Enum):
    """List available buttons."""

    A = 0
    B = 1


@ensure(lambda result: len(result.shape) == 2)
@ensure(lambda result: result.dtype == np.uint8)
def prepare_direction_rose() -> cv2.Mat:
    """
    Pre-compute the direction rose as a fixed map.

    Each pixel denotes the direction.
    """
    radius = 256
    w = 2 * radius
    h = w

    # The default is unknown direction.
    mat = 255 * np.zeros((w, h), np.uint8)

    origin_x = w / 2
    origin_y = h / 2

    idle_radius = 0.3 * radius

    idle_radius_square = idle_radius**2

    segment_angle = 360.0 / 8.0
    half_segment_angle = segment_angle / 2
    boundaries = [half_segment_angle + i * segment_angle for i in range(8)]
    indexed_directions = [
        Direction.EAST,  # corresponds to range ``[boundaries[-1], boundaries[0])``
        Direction.NORTH_EAST,
        Direction.NORTH,
        Direction.NORTH_WEST,
        Direction.WEST,
        Direction.SOUTH_WEST,
        Direction.SOUTH,
        Direction.SOUTH_EAST,
    ]

    for y in range(h):
        for x in range(w):
            distance_square = (x - origin_x) ** 2 + (y - origin_y) ** 2
            if distance_square < idle_radius_square:
                mat[y, x] = Direction.NOWHERE.value
            else:
                # NOTE (mristin, 2023-04-15):
                # We have to negate in y-direction since the coordinate system
                # in images starts at top-left and increases downwards.
                #
                # We compute here the angle relative to the x-axis as we draw it on
                # the paper.
                angle_rad = math.atan2(-(y - origin_y), x - origin_x)

                if angle_rad < 0:
                    angle_rad += 2 * math.pi

                angle_deg = angle_rad / (2 * math.pi) * 360.0

                if angle_deg >= boundaries[-1]:
                    direction_i = 0
                else:
                    direction_i = None

                    prev_boundary = 0.0
                    for i, boundary in enumerate(boundaries):
                        if prev_boundary <= angle_deg < boundary:
                            direction_i = i
                            break

                        prev_boundary = boundary

                assert direction_i is not None

                direction = indexed_directions[direction_i]

                mat[y, x] = direction.value

    assert (mat < 255).all()

    return mat


@require(lambda direction_rose: len(direction_rose.shape) == 2)
@ensure(lambda direction_rose, result: direction_rose.shape[0] == result.shape[0])
@ensure(lambda direction_rose, result: direction_rose.shape[1] == result.shape[1])
@ensure(lambda result: result.shape[2] == 4)
def colorize_direction_rose(direction_rose: cv2.Mat, direction: Direction) -> cv2.Mat:
    """Colorize the direction rose and highlight the selected one."""
    h, w = direction_rose.shape

    # The last channel is alpha.
    image = np.zeros((w, h, 4), dtype=np.uint8)

    # NOTE (mristin, 2023-04-15):
    # We define colors in RGB here.
    direction_value_to_color = {
        Direction.NOWHERE.value: (0, 0, 0),
        Direction.NORTH.value: (40, 40, 40),
        Direction.NORTH_EAST.value: (233, 83, 107),
        Direction.EAST.value: (97, 208, 79),
        Direction.SOUTH_EAST.value: (34, 151, 230),
        Direction.SOUTH.value: (40, 226, 229),
        Direction.SOUTH_WEST.value: (205, 11, 188),
        Direction.WEST.value: (245, 199, 16),
        Direction.NORTH_WEST.value: (158, 158, 158),
    }

    for y in range(h):
        for x in range(w):
            direction_value = direction_rose[y, x]
            red, green, blue = direction_value_to_color[direction_value]

            alpha = int(round(255 * 0.3))
            if direction.value == direction_value:
                alpha = int(round(255 * 0.7))

            # NOTE (mristin, 2023-04-15):
            # OpenCV works in BGR.
            image[y, x, 0] = blue
            image[y, x, 1] = green
            image[y, x, 2] = red
            image[y, x, 3] = alpha

    return image


@require(lambda rescaled_direction_rose: rescaled_direction_rose.shape[2] == 4)
def draw_direction_rose(
    canvas: cv2.Mat, center_xy: Tuple[float, float], rescaled_direction_rose: cv2.Mat
) -> None:
    """Draw the colorized direction rose at ``center_xy``."""
    rose_h, rose_w, _ = rescaled_direction_rose.shape
    canvas_h, canvas_w, _ = canvas.shape

    center_x, center_y = center_xy

    # Rose in the coordinates of the canvas
    xmin = int(round(center_x - rose_w / 2.0))
    ymin = int(round(center_y - rose_h / 2.0))
    xmax = xmin + rose_w
    ymax = ymin + rose_h

    # NOTE (mristin, 2023-04-15):
    # If we can not blit the canvas, simply return.
    if xmax < 0 or xmin >= canvas_w or ymax < 0 or ymin >= canvas_h:
        return

    if xmin < 0:
        xmin_rose = -xmin
    else:
        xmin_rose = 0

    if xmax > canvas_w:
        xmax_rose = canvas_w - xmin
    else:
        xmax_rose = rose_w

    if ymin < 0:
        ymin_rose = -ymin
    else:
        ymin_rose = 0

    if ymax > canvas_h:
        ymax_rose = canvas_h - ymin
    else:
        ymax_rose = rose_h

    # NOTE (mristin, 2023-04-15):
    # We finally clip xmin, ymin, xmax and ymax to fit the canvas as we do not
    # need the non-clipped values anymore.
    xmin = max(0, xmin)
    ymin = max(0, ymin)
    xmax = min(canvas_w, xmax)
    ymax = min(canvas_h, ymax)

    assert xmax_rose - xmin_rose == xmax - xmin, (
        f"{xmin=}, {xmax=}, {xmax - xmin=}, "
        f"{xmin_rose=}, {xmax_rose=}, {xmax_rose - xmin_rose=}"
    )

    assert ymax_rose - ymin_rose == ymax - ymin, (
        f"{ymin=}, {ymax=}, {ymax - ymin=}, "
        f"{ymin_rose=}, {ymax_rose=}, {ymax_rose - ymin_rose=}"
    )

    assert 0 <= xmin < canvas_w, f"{xmin=}, {canvas_w=}"
    assert 0 <= xmax <= canvas_w, f"{xmax=}, {canvas_w=}"
    assert 0 <= ymin < canvas_h, f"{ymin=}, {canvas_h=}"
    assert 0 <= ymax <= canvas_h, f"{ymax=}, {canvas_h=}"

    assert 0 <= xmin_rose < rose_w, f"{xmin_rose=}, {rose_w=}"
    assert 0 <= xmax_rose <= rose_w, f"{xmax_rose=}, {rose_w=}"
    assert 0 <= ymin_rose < rose_h, f"{ymin_rose=}, {rose_h=}"
    assert 0 <= ymax_rose <= rose_h, f"{ymax_rose=}, {rose_h=}"

    alpha = rescaled_direction_rose[ymin_rose:ymax_rose, xmin_rose:xmax_rose, 3] / 255.0
    one_minus_alpha = 1 - alpha

    canvas_crop = canvas[ymin:ymax, xmin:xmax, :]

    for channel in range(3):
        canvas[ymin:ymax, xmin:xmax, channel] = np.multiply(
            canvas_crop[..., channel], one_minus_alpha
        ) + (
            alpha
            * rescaled_direction_rose[ymin_rose:ymax_rose, xmin_rose:xmax_rose, channel]
        )


def determine_torso_size(
    detection: bodypose.Detection, frame_wh: Tuple[int, int]
) -> Optional[float]:
    """
    Determine the size of the torso in pixels.

    If torso is not visible, return None.
    """
    _, h = frame_wh

    left_shoulder = detection.keypoints.get(bodypose.KeypointLabel.LEFT_SHOULDER, None)
    right_shoulder = detection.keypoints.get(
        bodypose.KeypointLabel.RIGHT_SHOULDER, None
    )
    left_hip = detection.keypoints.get(bodypose.KeypointLabel.LEFT_HIP, None)
    right_hip = detection.keypoints.get(bodypose.KeypointLabel.RIGHT_HIP, None)

    if (
        left_shoulder is None
        or right_shoulder is None
        or left_hip is None
        or right_hip is None
    ):
        return None

    shoulder_y = (left_shoulder.y + right_shoulder.y) / 2.0
    hip_y = (left_hip.y + right_hip.y) / 2.0

    if shoulder_y > hip_y:
        return None

    return (hip_y * h - shoulder_y * h) * 1.5


def determine_center(
    detection: bodypose.Detection, hand: Hand, frame_wh: Tuple[int, int]
) -> Optional[Tuple[float, float]]:
    """
    Determine the shoulder point of the given hand and return its (x, y) in pixels.

    If the shoulder is not visible, return None.
    """
    w, h = frame_wh

    if hand is Hand.LEFT:
        # NOTE (mristin, 2023-04-15):
        # We flip the frame, so we have to take the *opposite* keypoint.
        left_shoulder = detection.keypoints.get(
            bodypose.KeypointLabel.RIGHT_SHOULDER, None
        )
        if left_shoulder is None:
            return None

        return left_shoulder.x * w, left_shoulder.y * h

    elif hand is Hand.RIGHT:
        # NOTE (mristin, 2023-04-15):
        # We flip the frame, so we have to take the *opposite* keypoint.
        right_shoulder = detection.keypoints.get(
            bodypose.KeypointLabel.LEFT_SHOULDER, None
        )
        if right_shoulder is None:
            return None

        return right_shoulder.x * w, right_shoulder.y * h
    else:
        common.assert_never(hand)


def determine_pointer(
    detection: bodypose.Detection, hand: Hand, frame_wh: Tuple[int, int]
) -> Optional[Tuple[float, float]]:
    """
    Determine the wrist point of the given hand and return its (x, y) in pixels.

    If the wrist is not visible, return None.
    """
    w, h = frame_wh

    if hand is Hand.LEFT:
        # NOTE (mristin, 2023-04-15):
        # We flip the frame, so we have to take the *opposite* keypoint.
        left_wrist = detection.keypoints.get(bodypose.KeypointLabel.RIGHT_WRIST, None)
        if left_wrist is None:
            return None
        return left_wrist.x * w, left_wrist.y * h

    elif hand is Hand.RIGHT:
        # NOTE (mristin, 2023-04-15):
        # We flip the frame, so we have to take the *opposite* keypoint.
        right_wrist = detection.keypoints.get(bodypose.KeypointLabel.LEFT_WRIST, None)
        if right_wrist is None:
            return None
        return right_wrist.x * w, right_wrist.y * h
    else:
        common.assert_never(hand)


def draw_center(center_xy: Tuple[float, float], canvas: cv2.Mat) -> None:
    """Mark the center of the arrows."""
    x, y = center_xy

    h, w, _ = canvas.shape

    x_int = max(0, min(w, int(round(x))))
    y_int = max(0, min(h, int(round(y))))

    cv2.circle(canvas, (x_int, y_int), 5, (255, 255, 255), -1)


def draw_pointer(pointer_xy: Tuple[float, float], canvas: cv2.Mat) -> None:
    """Mark the pointer to the arrows."""
    x, y = pointer_xy

    h, w, _ = canvas.shape

    x_int = max(0, min(w, int(round(x))))
    y_int = max(0, min(h, int(round(y))))

    cv2.circle(canvas, (x_int, y_int), 5, (255, 255, 255), -1)


def determine_direction(
    torso_size: float,
    center_xy: Tuple[float, float],
    pointer_xy: Tuple[float, float],
    direction_rose: cv2.Mat,
) -> Optional[Direction]:
    """Determine the direction from the center and pointer rescaled to torso."""
    rose_h, rose_w = direction_rose.shape

    cursor_x_frame = pointer_xy[0] - center_xy[0]
    cursor_y_frame = pointer_xy[1] - center_xy[1]

    cursor_x = rose_w / 2.0 + (cursor_x_frame / torso_size) * (rose_w / 2.0)
    cursor_y = rose_h / 2.0 + (cursor_y_frame / torso_size) * (rose_h / 2.0)

    cursor_x_int = int(round(cursor_x))
    cursor_y_int = int(round(cursor_y))

    if cursor_x_int < 0 or cursor_x_int >= rose_w:
        return None

    if cursor_y_int < 0 or cursor_y_int >= rose_h:
        return None

    direction_value = direction_rose[cursor_y_int, cursor_x_int]

    return DIRECTION_FROM_VALUE[direction_value]


def determine_ankle_and_knee_position(
    detection: bodypose.Detection, leg: Leg, frame_wh: Tuple[int, int]
) -> Optional[Tuple[Tuple[float, float], Tuple[float, float]]]:
    """Determine the ankle and knee position, if visible."""
    # NOTE (mristin, 2023-04-15):
    # We flip the frame, so we have to take the *opposite* keypoint.

    w, h = frame_wh

    if leg is Leg.LEFT:
        left_ankle = detection.keypoints.get(bodypose.KeypointLabel.RIGHT_ANKLE, None)
        if left_ankle is None:
            return None

        left_knee = detection.keypoints.get(bodypose.KeypointLabel.RIGHT_KNEE, None)
        if left_knee is None:
            return None

        return (
            (left_ankle.x * w, left_ankle.y * h),
            (left_knee.x * w, left_knee.y * h),
        )

    elif leg is Leg.RIGHT:
        right_ankle = detection.keypoints.get(bodypose.KeypointLabel.LEFT_ANKLE, None)
        if right_ankle is None:
            return None

        right_knee = detection.keypoints.get(bodypose.KeypointLabel.LEFT_KNEE, None)
        if right_knee is None:
            return None

        return (
            (right_ankle.x * w, right_ankle.y * h),
            (right_knee.x * w, right_knee.y * h),
        )

    else:
        common.assert_never(leg)


def determine_legs_pressed(
    left_ankle_xy: Tuple[float, float],
    left_knee_xy: Tuple[float, float],
    right_ankle_xy: Tuple[float, float],
    right_knee_xy: Tuple[float, float],
) -> Tuple[bool, bool]:
    """Determine which leg are pressed (up towards the knee)."""
    # NOTE (mristin, 2023-04-16):
    # Remember that the image coordinates start at top-left corner.

    # region Left
    left_pressed = False

    left_ratio = (right_ankle_xy[1] - left_ankle_xy[1]) / (
        right_ankle_xy[1] - right_knee_xy[1]
    )
    if left_ratio > 0.2:
        left_pressed = True

    # endregion

    # region Right

    right_pressed = False

    right_ratio = (left_ankle_xy[1] - right_ankle_xy[1]) / (
        left_ankle_xy[1] - left_knee_xy[1]
    )
    if right_ratio > 0.2:
        right_pressed = True

    # endregion

    return left_pressed, right_pressed


def draw_ankles(
    left_ankle_xy: Tuple[float, float],
    right_ankle_xy: Tuple[float, float],
    left_leg_button: Button,
    right_leg_button: Button,
    left_leg_pressed: bool,
    right_leg_pressed: bool,
    canvas: cv2.Mat,
) -> None:
    """Draw the state of the legs."""

    h, w, _ = canvas.shape

    left_leg_letter = ""
    if left_leg_button is Button.A:
        left_leg_letter = "A"
    elif left_leg_button is Button.B:
        left_leg_letter = "B"
    else:
        common.assert_never(left_leg_button)

    right_leg_letter = ""
    if right_leg_button is Button.A:
        right_leg_letter = "A"
    elif right_leg_button is Button.B:
        right_leg_letter = "B"
    else:
        common.assert_never(right_leg_button)

    # region Left

    # NOTE (mristin, 2023-04-16):
    # The order of drawings matters for the legibility!

    left_ankle_xy_int = (
        max(0, min(w, int(round(left_ankle_xy[0])))),
        max(0, min(h, int(round(left_ankle_xy[1])))),
    )

    cv2.circle(
        canvas,
        left_ankle_xy_int,
        (5 if not left_leg_pressed else 20),
        (0, 255, 255),
        -1,
    )

    cv2.putText(
        canvas,
        left_leg_letter,
        left_ankle_xy_int,
        cv2.FONT_HERSHEY_PLAIN,
        3.0,
        (255, 255, 255),
        4,
        cv2.LINE_AA,
    )

    # endregion

    # region Right

    right_ankle_xy_int = (
        max(0, min(w, int(round(right_ankle_xy[0])))),
        max(0, min(h, int(round(right_ankle_xy[1])))),
    )

    cv2.circle(
        canvas,
        right_ankle_xy_int,
        (5 if not right_leg_pressed else 20),
        (255, 255, 0),
        -1,
    )

    cv2.putText(
        canvas,
        right_leg_letter,
        right_ankle_xy_int,
        cv2.FONT_HERSHEY_PLAIN,
        3.0,
        (255, 255, 255),
        4,
        cv2.LINE_AA,
    )

    # endregion


@require(lambda canvas: canvas.shape[2] == 3)
def draw_instructions(canvas: cv2.Mat) -> cv2.Mat:
    """Enlarge the canvas and write the instructions at the bottom."""
    h, w, _ = canvas.shape

    new_canvas = np.zeros((h + 50, w, 3), dtype=np.uint8)

    new_canvas[0:h, 0:w, 0] = canvas[..., 0]
    new_canvas[0:h, 0:w, 1] = canvas[..., 1]
    new_canvas[0:h, 0:w, 2] = canvas[..., 2]

    cv2.putText(
        new_canvas,
        "Press 'Q' to quit, 'S' to switch shoulder, 'L' to switch legs",
        (10, h + 40),
        cv2.FONT_HERSHEY_COMPLEX_SMALL,
        0.7,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )

    return new_canvas


def main(prog: str) -> int:
    """
    Execute the main routine.

    :param prog: name of the program to be displayed in the help
    :return: exit code
    """
    parser = argparse.ArgumentParser(prog=prog, description=__doc__)
    parser.add_argument(
        "--version", help="show the current version and exit", action="store_true"
    )
    parser.add_argument(
        "--camera_index",
        help=(
            "Index for the camera that should be used. Usually 0 is your web cam, "
            "but there are also systems where the web cam was given at index -1 or 2. "
            "We rely on OpenCV and this has not been fixed in OpenCV yet. Please see "
            "https://github.com/opencv/opencv/issues/4269"
        ),
        default=0,
        type=int,
    )
    parser.add_argument(
        "--hotkey_for_north", help="Map direction to the hotkey", default="up"
    )
    parser.add_argument(
        "--hotkey_for_north_east",
        help="Map direction to the hotkey",
        default="up+right",
    )
    parser.add_argument(
        "--hotkey_for_east", help="Map direction to the hotkey", default="right"
    )
    parser.add_argument(
        "--hotkey_for_south_east",
        help="Map direction to the hotkey",
        default="down+right",
    )
    parser.add_argument(
        "--hotkey_for_south", help="Map direction to the hotkey", default="down"
    )
    parser.add_argument(
        "--hotkey_for_south_west", help="Map direction to the hotkey", default="up+left"
    )
    parser.add_argument(
        "--hotkey_for_west", help="Map direction to the hotkey", default="left"
    )
    parser.add_argument(
        "--hotkey_for_north_west", help="Map direction to the hotkey", default="up+left"
    )
    parser.add_argument(
        "--hotkey_for_button_a", help="Map button to the hotkey", default="space"
    )
    parser.add_argument(
        "--hotkey_for_button_b", help="Map button to the hotkey", default="ctrl"
    )

    # NOTE (mristin, 2022-12-16):
    # The module ``argparse`` is not flexible enough to understand special options such
    # as ``--version`` so we manually hard-wire.
    if "--version" in sys.argv and "--help" not in sys.argv:
        print(bodygamepad.__version__)
        return 0

    args = parser.parse_args()

    camera_index = int(args.camera_index)

    # NOTE (mristin, 2023-04-16):
    # We need a smarter approach to pressing and releasing keys in case the hotkeys
    # intersect between the legs and directions.

    direction_to_hotkey = {
        Direction.NORTH: args.hotkey_for_north,
        Direction.NORTH_EAST: args.hotkey_for_north_east,
        Direction.EAST: args.hotkey_for_east,
        Direction.SOUTH_EAST: args.hotkey_for_south_east,
        Direction.SOUTH: args.hotkey_for_south,
        Direction.SOUTH_WEST: args.hotkey_for_south_west,
        Direction.WEST: args.hotkey_for_west,
        Direction.NORTH_WEST: args.hotkey_for_north_west,
    }

    button_to_hotkey = {
        Button.A: args.hotkey_for_button_a,
        Button.B: args.hotkey_for_button_b,
    }

    print("Preparing the direction rose...")
    direction_rose = prepare_direction_rose()

    print("Caching different direction activations...")
    direction_to_colorized_rose = {
        direction: colorize_direction_rose(
            direction_rose=direction_rose, direction=direction
        )
        for direction in Direction
    }

    print("Loading the detector...")

    # noinspection SpellCheckingInspection
    detector = bodypose.load_detector(
        PACKAGE_DIR / "media" / "models" / "312f001449331ee3d410d758fccdc9945a65dbc3"
    )

    cap = None  # type: Optional[cv2.VideoCapture]

    print("Opening the video capture...")
    try:
        cap = cv2.VideoCapture(camera_index)

    except Exception as exception:
        print(
            f"Failed to open the video capture at index {camera_index}: {exception}",
            file=sys.stderr,
        )
        return 1

    hand = Hand.RIGHT

    left_leg_button = Button.A
    right_leg_button = Button.B

    prev_direction = Direction.NOWHERE

    prev_left_leg_pressed = False
    prev_right_leg_pressed = False

    cv2.namedWindow("body-gamepad", cv2.WINDOW_NORMAL)

    try:
        while True:
            reading_ok, frame = cap.read()
            if not reading_ok:
                print("Failed to read a frame from the video capture.", file=sys.stderr)
                break

            frame = cv2.flip(frame, 1)

            h, w, _ = frame.shape
            frame_wh = (w, h)

            direction = None  # type: Optional[Direction]
            left_leg_pressed = False
            right_leg_pressed = False

            detections = detector(frame)
            if len(detections) > 0:
                detection = detections[0]

                # region Direction
                torso_size = determine_torso_size(
                    detection=detection, frame_wh=frame_wh
                )
                center_xy = determine_center(
                    detection=detection, hand=hand, frame_wh=frame_wh
                )
                pointer_xy = determine_pointer(
                    detection=detection, hand=hand, frame_wh=frame_wh
                )

                if center_xy is not None:
                    draw_center(center_xy=center_xy, canvas=frame)

                if pointer_xy is not None:
                    draw_pointer(pointer_xy=pointer_xy, canvas=frame)

                if (
                    torso_size is not None
                    and center_xy is not None
                    and pointer_xy is not None
                ):
                    direction = determine_direction(
                        torso_size=torso_size,
                        center_xy=center_xy,
                        pointer_xy=pointer_xy,
                        direction_rose=direction_rose,
                    )

                    if direction is not None:
                        colorized_direction_rose = direction_to_colorized_rose[
                            direction
                        ]
                        rescaled_direction_rose = cv2.resize(
                            colorized_direction_rose,
                            (int(2.0 * torso_size), int(2.0 * torso_size)),
                        )

                        draw_direction_rose(
                            canvas=frame,
                            center_xy=center_xy,
                            rescaled_direction_rose=rescaled_direction_rose,
                        )
                # endregion

                left_ankle_and_knee = determine_ankle_and_knee_position(
                    detection=detection, leg=Leg.LEFT, frame_wh=frame_wh
                )

                right_ankle_and_knee = determine_ankle_and_knee_position(
                    detection=detection, leg=Leg.RIGHT, frame_wh=frame_wh
                )

                if left_ankle_and_knee is not None and right_ankle_and_knee is not None:
                    left_ankle_xy, left_knee_xy = left_ankle_and_knee
                    right_ankle_xy, right_knee_xy = right_ankle_and_knee

                    left_leg_pressed, right_leg_pressed = determine_legs_pressed(
                        left_ankle_xy=left_ankle_xy,
                        left_knee_xy=left_knee_xy,
                        right_ankle_xy=right_ankle_xy,
                        right_knee_xy=right_knee_xy,
                    )

                    draw_ankles(
                        left_ankle_xy=left_ankle_xy,
                        right_ankle_xy=right_ankle_xy,
                        left_leg_button=left_leg_button,
                        right_leg_button=right_leg_button,
                        left_leg_pressed=left_leg_pressed,
                        right_leg_pressed=right_leg_pressed,
                        canvas=frame,
                    )

            if direction is None:
                direction = Direction.NOWHERE

            if direction is Direction.NOWHERE and prev_direction is Direction.NOWHERE:
                # NOTE (mristin, 2023-04-16):
                # There is no key to be pressed.
                pass

            elif (
                direction is Direction.NOWHERE
                and prev_direction is not Direction.NOWHERE
            ):
                # NOTE (mristin, 2023-04-16):
                # We have to release the key.
                prev_hotkey = direction_to_hotkey[prev_direction]
                keyboard.release(prev_hotkey)

            elif (
                direction is not Direction.NOWHERE
                and prev_direction is Direction.NOWHERE
            ):
                # NOTE (mristin, 2023-04-16):
                # We press the key.
                hotkey = direction_to_hotkey[direction]
                keyboard.press(hotkey)

            elif (
                direction is not Direction.NOWHERE
                and prev_direction is not Direction.NOWHERE
                and direction is prev_direction
            ):
                # NOTE (mristin, 2023-04-16):
                # Leave the keys pressed.
                pass

            elif (
                direction is not Direction.NOWHERE
                and prev_direction is not Direction.NOWHERE
                and direction is not prev_direction
            ):
                # NOTE (mristin, 2023-04-16):
                # We release the previous key and press a new one.
                prev_hotkey = direction_to_hotkey[prev_direction]
                keyboard.release(prev_hotkey)

                hotkey = direction_to_hotkey[direction]
                keyboard.press(hotkey)

            else:
                raise AssertionError(
                    f"Unhandled execution path: {direction=}, {prev_direction=}"
                )

            prev_direction = direction

            left_leg_hotkey = button_to_hotkey[left_leg_button]
            right_leg_hotkey = button_to_hotkey[right_leg_button]

            if left_leg_pressed and not prev_left_leg_pressed:
                keyboard.press(left_leg_hotkey)
            elif not left_leg_pressed and prev_left_leg_pressed:
                keyboard.release(left_leg_hotkey)
            else:
                # NOTE (mristin, 2023-04-16):
                # Leave pressed or released as before.
                pass

            if right_leg_pressed and not prev_right_leg_pressed:
                keyboard.press(right_leg_hotkey)
            elif not right_leg_pressed and prev_right_leg_pressed:
                keyboard.release(right_leg_hotkey)
            else:
                # NOTE (mristin, 2023-04-16):
                # Leave pressed or released as before.
                pass

            prev_left_leg_pressed = left_leg_pressed
            prev_right_leg_pressed = right_leg_pressed

            frame = draw_instructions(frame)

            cv2.imshow("body-gamepad", frame)
            key = cv2.waitKey(10) & 0xFF

            if key == ord("q"):
                print("Received 'q', quitting...")
                break

            elif key == ord("s"):
                if hand is Hand.LEFT:
                    hand = Hand.RIGHT
                elif hand is Hand.RIGHT:
                    hand = Hand.LEFT
                else:
                    common.assert_never(hand)

            elif key == ord("l"):
                left_leg_button, right_leg_button = right_leg_button, left_leg_button
            else:
                pass

    finally:
        if cap is not None:
            print("Closing the video capture...")
            cap.release()
            print("Video capture closed.")

    print("Goodbye.")

    return 0


def entry_point() -> int:
    """Provide an entry point for a console script."""
    return main(prog="body-gamepad")


if __name__ == "__main__":
    sys.exit(main(prog="body-gamepad"))

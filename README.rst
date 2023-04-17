************
body-gamepad
************

Emulate arrow keys and two buttons with your arms and legs.

.. image:: https://github.com/mristin/body-gamepad/actions/workflows/ci.yml/badge.svg
    :target: https://github.com/mristin/body-gamepad/actions/workflows/ci.yml
    :alt: Continuous integration

For example, you can play DOS games on your computer such as Bubble Bobble:

.. image:: https://media.githubusercontent.com/media/mristin/body-gamepad/main/bubble-bobble.gif
    :alt: Bubble Bobble

Or Gameboy-emulated Super Mario:

.. image:: https://media.githubusercontent.com/media/mristin/body-gamepad/main/super-mario.gif
    :alt: Super Mario on Gameboy

Or practice bass guitar while you play Tetris:

.. image:: https://media.githubusercontent.com/media/mristin/body-gamepad/main/tetris.gif
    :alt: Tetris

Or practice dribbling in basketball while skiing-or-dying:

.. image:: https://media.githubusercontent.com/media/mristin/body-gamepad/main/ski-or-die.gif
    :alt: Ski-or-die

Installation
============
Download and unzip a version of the game from the `Releases`_.

.. _Releases: https://github.com/mristin/body-gamepad/releases

Running
=======
Run ``body-gamepad.exe`` (in the directory where you unzipped the program).


``--help``
==========

.. Help starts: body-gamepad.exe --help
.. code-block::

    usage: body-gamepad [-h] [--version] [--camera_index CAMERA_INDEX]
                        [--hotkey_for_north HOTKEY_FOR_NORTH]
                        [--hotkey_for_north_east HOTKEY_FOR_NORTH_EAST]
                        [--hotkey_for_east HOTKEY_FOR_EAST]
                        [--hotkey_for_south_east HOTKEY_FOR_SOUTH_EAST]
                        [--hotkey_for_south HOTKEY_FOR_SOUTH]
                        [--hotkey_for_south_west HOTKEY_FOR_SOUTH_WEST]
                        [--hotkey_for_west HOTKEY_FOR_WEST]
                        [--hotkey_for_north_west HOTKEY_FOR_NORTH_WEST]
                        [--hotkey_for_button_a HOTKEY_FOR_BUTTON_A]
                        [--hotkey_for_button_b HOTKEY_FOR_BUTTON_B]

    Emulate arrow keys and two buttons with your arms and legs.

    optional arguments:
      -h, --help            show this help message and exit
      --version             show the current version and exit
      --camera_index CAMERA_INDEX
                            Index for the camera that should be used. Usually 0 is
                            your web cam, but there are also systems where the web
                            cam was given at index -1 or 2. We rely on OpenCV and
                            this has not been fixed in OpenCV yet. Please see
                            https://github.com/opencv/opencv/issues/4269
      --hotkey_for_north HOTKEY_FOR_NORTH
                            Map direction to the hotkey
      --hotkey_for_north_east HOTKEY_FOR_NORTH_EAST
                            Map direction to the hotkey
      --hotkey_for_east HOTKEY_FOR_EAST
                            Map direction to the hotkey
      --hotkey_for_south_east HOTKEY_FOR_SOUTH_EAST
                            Map direction to the hotkey
      --hotkey_for_south HOTKEY_FOR_SOUTH
                            Map direction to the hotkey
      --hotkey_for_south_west HOTKEY_FOR_SOUTH_WEST
                            Map direction to the hotkey
      --hotkey_for_west HOTKEY_FOR_WEST
                            Map direction to the hotkey
      --hotkey_for_north_west HOTKEY_FOR_NORTH_WEST
                            Map direction to the hotkey
      --hotkey_for_button_a HOTKEY_FOR_BUTTON_A
                            Map button to the hotkey
      --hotkey_for_button_b HOTKEY_FOR_BUTTON_B
                            Map button to the hotkey

.. Help ends: body-gamepad.exe --help

Acknowledgments
===============
The model has been downloaded from TensorFlow Hub: https://tfhub.dev/google/movenet/multipose/lightning/1

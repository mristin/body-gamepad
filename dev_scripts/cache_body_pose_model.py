"""Cache the bodypose model in the resource directory."""

import argparse
import os
import pathlib
import sys

import bodygamepad.bodypose

def main() -> int:
    """Execute the main routine."""
    this_path = pathlib.Path(os.path.realpath(__file__))
    os.environ["TFHUB_CACHE_DIR"] = str(
    this_path.parent.parent / "bodygamepad" / "media" / "models")

    print("Loading the detector...")
    _ = bodygamepad.bodypose.load_detector()
    print("Loaded.")

    return 0


if __name__ == "__main__":
    sys.exit(main())

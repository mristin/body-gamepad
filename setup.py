"""
A setuptools based setup module.

See:
https://packaging.python.org/en/latest/distributing.html
https://github.com/pypa/sampleproject
"""

import os
import sys

from setuptools import setup, find_packages

# pylint: disable=redefined-builtin

setup(
    name="body-gamepad",
    # Don't forget to update the version in __init__.py and CHANGELOG.rst!
    version="0.0.1",
    description="Emulate arrow keys and two buttons with your arms and legs.",
    url="https://github.com/mristin/body-gamepad",
    author="Marko Ristin",
    author_email="marko@ristin.ch",
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Education",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.9",
    ],
    license="License :: OSI Approved :: MIT License",
    keywords="body pose estimation arrow keys buttons augmented reality",
    install_requires=[
        "opencv-python>=4,<5",
        "tensorflow==2.12.0",
        "icontract>=2.6.1,<3",
        "keyboard==0.13.5",
    ],
    extras_require={
        "dev": [
            "black==23.3.0",
            "mypy==1.2.0",
            "pylint==2.17.2",
            "coverage>=6.5.0,<7",
            "pyinstaller>=5,<6",
            "tensorflow-hub==0.13.0",
        ],
    },
    py_modules=["bodygamepad"],
    packages=find_packages(exclude=["tests", "continuous_integration", "dev_scripts"]),
    package_data={
        "bodygamepad": [
            "media/models/*",
        ]
    },
    entry_points={
        "console_scripts": [
            "body-gamepad=bodygamepad.main:entry_point",
        ]
    },
)

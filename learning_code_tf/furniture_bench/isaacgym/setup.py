#   DEPRECATION: Legacy editable install of isaacgym==1.0rc4 from file:///ssddata/qtguo/GENERAL_DATA/isaacgym/python (setup.py develop) is deprecated. pip 25.1 will enforce this behaviour change. A possible replacement is to add a pyproject.toml or enable --use-pep517, and use setuptools >= 64. If the resulting installation is not behaving as expected, try using --config-settings editable_mode=compat. Please consult the setuptools documentation for more information. Discussion can be found at https://github.com/pypa/pip/issues/11457
#   Running setup.py develop for isaacgym
# ERROR: pip's dependency resolver does not currently take into account all the packages that are installed. This behaviour is the source of the following dependency conflicts.
# torchvision 0.21.0 requires torch==2.6.0, which is not installed.
# robomimic 0.3.0 requires torch, which is not installed.
# numba 0.61.0 requires numpy<2.2,>=1.24, but you have numpy 2.2.2 which is incompatible.
# matplotlib 3.7.5 requires numpy<2,>=1.20, but you have numpy 2.2.2 which is incompatible.
# dppo 0.6.0 requires imageio==2.35.1, but you have imageio 2.37.0 which is incompatible.
# tensorflow 2.17.0 requires numpy<2.0.0,>=1.26.0; python_version >= "3.12", but you have numpy 2.2.2 which is incompatible.
# Successfully installed imageio-2.37.0 isaacgym-1.0rc4 ninja-1.11.1.3 numpy-2.2.2 pillow-11.1.0 pyyaml-6.0.2 scipy-1.15.1

"""Setup script for isaacgym"""

import sys
import os

from setuptools import setup, find_packages

def collect_files(target_dir):
    file_list = []
    for (root, dirs, files) in os.walk(target_dir,followlinks=True):
        for filename in files:
            file_list.append(os.path.join('..', root, filename))
    return file_list

def _do_setup():
    root_dir = os.path.dirname(os.path.realpath(__file__))

    packages = find_packages(".")
    print(packages)

    #
    # TODO: do something more clever to collect only the bindings for the active versions of Python
    #

    package_files = []
    if sys.platform.startswith("win"):
        package_files = package_files + collect_files("isaacgym/_bindings/windows-x86_64")
    elif sys.platform.startswith("linux"):
        package_files = package_files + collect_files("isaacgym/_bindings/linux-x86_64")

    setup(name='isaacgym',
          version='1.0.preview4',
          description='GPU-accelerated simulation and reinforcement learning toolkit',
          author='NVIDIA CORPORATION',
          author_email='',
          url='http://developer.nvidia.com/isaac-gym',
          license='Copyright (c) 2020-2022, NVIDIA CORPORATION.  All rights reserved.',
          packages=packages,
          package_data={
              "isaacgym": package_files
          },
          install_requires = [
              "numpy>=1.16.4",
              "scipy>=1.5.0",
              "pyyaml>=5.3.1",
              "pillow",
              "imageio",
              "ninja",
          ],
         )

_do_setup()



import re
import sys
from setuptools import find_packages, setup

with open("src/tissuemosaic/__init__.py") as f:
    for line in f:
        match = re.match('^__version__ = "(.*)"$', line)
        if match:
            __version__ = match.group(1)
            break

try:
    long_description = open("README.md", encoding="utf-8").read()
except Exception as e:
    sys.stderr.write("Failed to read README.md: {}\n".format(e))
    sys.stderr.flush()
    long_description = ""

setup(
    name="tissuemosaic",
    version=__version__,
    description="A Python library for the analysis of biological tissue and \
    cellular micron-environments based on self supervised learning",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=find_packages(where='src'),
    package_dir={'': 'src'},
    url="https://github.com/broadinstitute/TissueMosaic",
    author="Luca Dalessio, Sandeep Kambhampati",
    author_email="skambham@broadinstitute.org",
    # install_requires=[
    #     "anndata>=0.8.0",
    #     "leidenalg>=0.8.3",
    #     "lightly>=1.2.5",
    #     "lightning_bolts>=0.3.4",
    #     "matplotlib>=3.2.2",
    #     "neptune_client>=0.9.19",
    #     "numpy>=1.21",
    #     "pandas>=1.3.4",
    #     "protobuf~=3.19.4",
    #     "python_igraph>=0.9.9",
    #     "PyYAML>=6.0",
    #     "scanpy>=1.8.2",
    #     "scikit_learn>=1.0.2",
    #     "scipy>=1.6.2",
    #     "seaborn>=0.11.1",
    #     "torch>=1.10",
    #     "torchvision>=0.10.1",
    #     "umap_learn>=0.5.1"
    # ],
    # extras_require={
    #     "test": [
    #         "isort>=5.9",
    #         "flake8",
    #         "pytest>=5.0",
    #     ],
    # },
    python_requires=">=3.9",
    keywords="self supervised learning, tissue analysis",
    license="Apache 2.0",
    classifiers=[
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: POSIX :: Linux",
        "Operating System :: MacOS :: MacOS X",
        "Programming Language :: Python :: 3.9",
    ],
)

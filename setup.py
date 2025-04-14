import sys
import re

try:
    from importlib import util
except Exception:
    sys.stdout.write("\nIt seems that importlib library is not available on this machine. Please install pip (e.g. for Ubuntu, run 'sudo apt-get install python3-pip'.\n")
    sys.exit()

if util.find_spec("setuptools") is None:
    sys.stdout.write("\nIt seems that setuptools is not available on this machine. Please install pip (e.g. for Ubuntu, run 'sudo apt-get install python3-pip'.\n")
    sys.exit()

from setuptools import setup, find_packages

# Check Python version
if sys.version_info.major != 3 or sys.version_info.minor != 10:
    sys.exit("Sorry, Python == 3.10.x is the only supported version.")

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

# Set the software version number
VERSION = '1.0.0'
assert re.match(r"^[0-9]+\.[0-9]+\.[0-9]+$", VERSION), "Invalid version number"

setup(
    name='netmd',
    version=VERSION,
    author="Manuel Mangoni, Michele Pieroni",
#    author_email="bioinformatics@css-mendel.it",
    description="SOFTWARE DESCRIPTION HERE",
    url="https://github.com/mazzalab/netmd",
    packages=find_packages(),
    include_package_data=True,
    license="MIT",
    long_description=long_description,
    long_description_content_type="text/markdown",
    entry_points={"console_scripts": ["netmd = netmd.main:cli"]},
    install_requires=["setuptools"],
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Science/Research",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    project_urls={
        # 'Documentation': 'http://pyntacle.css-mendel.it:10080/#docs',
        "Source": "https://github.com/mazzalab/netmd",
        "Tracker": "https://github.com/mazzalab/netmd/issues",
    },
    keywords="network, embeddings, md, bioinformatics",
    python_requires="==3.10.14"
)

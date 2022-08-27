"""A setuptools based setup module.
See:
https://packaging.python.org/guides/distributing-packages-using-setuptools/
https://github.com/pypa/sampleproject
"""

from setuptools import setup, find_packages
import pathlib

here = pathlib.Path(__file__).parent.resolve()

# Arguments marked as "Required" below must be included for upload to PyPI.
# Fields marked as "Optional" may be commented out.

setup(
    # $ pip install sampleproject
    # And where it will live on PyPI: https://pypi.org/project/sampleproject/
    # There are some restrictions on what makes a valid project name
    # specification here:
    # https://packaging.python.org/specifications/core-metadata/#name
    name="afm-aux",
    version="1.0.0",
    description="tools for afm - mostly force spectroscopy",
    url="https://github.com/mmccraw98/afm",
    author="marshall mccraw",
    author_email="mmccraw98@gmail.com",
    keywords="atomicforcemicroscopy, afm, physics, data",
    # When your source code is in a subdirectory under the project root, e.g.
    # `src/`, it is necessary to specify the `package_dir` argument.
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    python_requires=">=3.7, <4",
    install_requires=["igor", "numpy", "matplotlib", "pandas", "scipy"],
    extras_require={
        "dev": ["check-manifest"],
        "test": ["coverage"],
    }
)
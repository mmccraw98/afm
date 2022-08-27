from setuptools import setup, find_packages
import pathlib

here = pathlib.Path(__file__).parent.resolve()

setup(
    name="afm-fd",
    version="1.0.0",
    description="tools for afm - mostly force spectroscopy",
    url="https://github.com/mmccraw98/afm",
    author="marshall mccraw",
    author_email="mmccraw98@gmail.com",
    keywords="atomicforcemicroscopy, afm, physics, data",
    # packages=find_packages(),
    packages=find_packages(),
    python_requires=">=3.7, <4",
    install_requires=["igor", "numpy", "matplotlib", "pandas", "scipy"],
    extras_require={
        "dev": ["check-manifest"],
        "test": ["coverage"],
    }
)

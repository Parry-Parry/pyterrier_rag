import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="pyterrier_rag",
    version="0.0.1",
    author="Andrew Parry",
    author_email="a.parry.1@research.gla.ac.uk",
    description="PyTerrier RAG: Retrieve and Generate",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.10',
)
import setuptools

with open("README.md", "r") as f:
    long_description = f.read()

package_data = {
    "": ["*.cpp", "*.cu"],
}

setuptools.setup(
    name="colbert-ai",
    version="0.2.20",
    author="Omar Khattab",
    author_email="okhattab@stanford.edu",
    description="Efficient and Effective Passage Search via Contextualized Late Interaction over BERT",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/stanford-futuredata/ColBERT",
    packages=setuptools.find_packages(),
    python_requires=">=3.8",
    install_requires=[
        "bitarray",
        "datasets",
        "flask",
        "GitPython",
        "python-dotenv",
        "ninja",
        "scipy",
        "tqdm",
        "transformers",
        "ujson",
    ],
    extras_require={
        "faiss-gpu": ["faiss-gpu>=1.7.0"],
        "faiss-cpu": ["faiss-cpu>=1.7.0"],
        "torch": ["torch==1.13.1"],
    },
    include_package_data=True,
    package_data=package_data,
)

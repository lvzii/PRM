# coding=utf-8
from setuptools import find_packages, setup

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

extras = {}
# extras["quality"] = ["ruff", "isort"]
# extras["tests"] = ["pytest"]
# extras["dev"] = ["vllm==0.6.3"] + extras["quality"] + extras["tests"]


install_requires = [
    # "accelerate",
    # "pebble",  # for parallel processing
    # "latex2sympy2==1.9.1",  # for MATH answer parsing
    # "word2number",  # for MATH answer parsing
    # "transformers>=4.47.0",
    # "fastapi",
]

setup(
    name="prm",
    version="0.1.0",
    author="youshuji",
    author_email="zjs20001205@gmail.com",
    description="A tool for prm methods on llms",
    long_description=open("README.md", "r", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/lvzii/prm",
    keywords="nlp llm prm",
    license="Apache",
    package_dir={"": "src"},
    packages=find_packages("src"),
    classifiers=[
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.10.9",
    install_requires=install_requires,
    extras_require=extras,
    include_package_data=True,
)

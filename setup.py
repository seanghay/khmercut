import setuptools

with open("README.md", "r") as f:
    long_description = f.read()

extras_requirements = {
    "test": [
        "pytest",
        "coverage",
    ],
}

setuptools.setup(
    name="khmercut",
    version="0.1.0",
    description="A (fast) Khmer word segmentation toolkit.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/seanghay/khmercut",
    author="Seanghay Yath",
    author_email="seanghay.dev@gmail.com",
    license="Apache License 2.0",
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
        "Intended Audience :: Developers",
        "Natural Language :: English",
    ],
    python_requires=">3.5",
    packages=setuptools.find_packages(),
    package_dir={"khmercut": "khmercut"},
    package_data={
        "khmercut": [
            "crf_ner_10000.crfsuite",
        ]
    },
    include_package_data=True,
    install_requires=["python-crfsuite"],
    extras_require=extras_requirements,
)

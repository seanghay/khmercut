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
    version="0.0.2",
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
    packages=setuptools.find_packages(exclude=["data", "bin", "samples", "scripts"]),
    package_dir={"khmercut": "khmercut"},
    package_data={
        "khmercut": [
          "crf_ner_10000.crfsuite",
        ]
    },
    include_package_data=True,
    install_requires=[
        "python-crfsuite==0.9.9",
        "khmernormalizer==0.0.4",
        "tqdm==4.65.0"
    ],
    extras_require=extras_requirements,
    scripts=['bin/khmercut'],
)
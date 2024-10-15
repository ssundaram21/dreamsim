import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()
long_description = "".join(long_description.split("<!--Experiments-->")[::2])
long_description = "".join(long_description.split("![teaser](images/figs/teaser.png)"))

setuptools.setup(
    name="dreamsim",
    version="0.2.1",
    description="DreamSim similarity metric",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/ssundaram21/dreamsim",
    packages=['dreamsim', 'dreamsim/feature_extraction'],
    install_requires=[
        "numpy",
        "open-clip-torch",
        "peft",
        "Pillow",
        "torch",
        "timm",
        "scipy",
        "torchvision",
        "transformers"
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)

# "scipy==1.9.2",
# "timm==0.6.12",
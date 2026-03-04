from setuptools import setup

setup(
    name="lazytorchtools",
    version="0.1.0",
    description="Tiny PyTorch utilities: small model builders and analysis helpers (PyTorch 2.0+)",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/deoliveirajoshua/lazytorchtools",
    py_modules=["lazytorchtools"],
    python_requires=">=3.8",
    install_requires=[
        "torch>=2.0",
        "numpy",
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
)

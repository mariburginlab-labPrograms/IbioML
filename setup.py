from setuptools import setup, find_packages

setup(
    name="ibioml",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "scikit-learn",
        "torch",
        "matplotlib",
        "scipy",
        "seaborn"
    ],
    author="Juani",
    description="ML toolkit for neuro decoding experiments at IBIoBA",
    include_package_data=True,
)

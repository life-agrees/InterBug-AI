from setuptools import setup, find_packages

setup(
    name="Sei_Module",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        'pandas',
        'requests',
        'scikit-learn'
    ],
)
from setuptools import setup, find_packages

setup(
    name='DCNN_SHHS',
    packages=find_packages(),
    version='0.1.0',
    description='Dialated Convolutional Neural Network for sleep stage classification on SHHS data',
    install_requires=['numpy', 'pytorch', 'h5py']
)

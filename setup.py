# Author: bbrighttaer
# Project: soek
# Date: 30/08/2019
# Time: 
# File: setup.py.py


from setuptools import setup

setup(
    name='soek',
    version='0.0.1',
    packages=['demo', 'soek'],
    url='',
    license='MIT',
    author='Brighter Agyemang',
    author_email='brighteragyemang@gmail.com',
    description='A simple hyperparameter searching library for Machine Learning.', install_requires=['numpy', 'skopt']
)

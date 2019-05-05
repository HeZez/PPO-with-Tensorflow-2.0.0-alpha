from setuptools import setup
from os import path

here = path.abspath(path.dirname(__file__))

setup(
    name='Setup for TF2 and ML AGENTS',
    version='0.1',
    description='Reinforcement Learning Setup',
    author='jw1401',
    
    classifiers=[],
    packages=[],  
    zip_safe=False,

    install_requires=[
        'mlagents_envs==0.8.1',
        'tensorflow==2.0.0-alpha0',
        'gym',
        'tensorboard',
        'flake8',
        'pylint',
        'pyaml',
        'click'],
    python_requires=">=3.5,<3.8",
)
from setuptools import setup, find_packages

setup(
    name='ReactSolver',
    version='0.1.0',
    description='A package for solving ODEs in chemical reactor design',
    author='Your Name',
    author_email='your.email@example.com',
    packages=find_packages(),
    install_requires=[
        'numpy',
        'scipy',
    ],
)

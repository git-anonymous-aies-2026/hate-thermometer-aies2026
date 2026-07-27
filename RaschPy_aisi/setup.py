from setuptools import setup, find_packages

setup(
    name='RaschPy',
    version='0.1.0',
    packages=find_packages(),
    install_requires=[
        'numpy',
        'pandas',
        'scipy',
        'matplotlib',
        'seaborn'
    ],
    python_requires='>=3.7',
)

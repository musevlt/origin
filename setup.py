from setuptools import setup, find_packages

setup(
    name='origin',
    version='1.0',
    packages=find_packages(),
    package_data={'origin': ['mumdatMask_1x1/*.fits.gz']},
    zip_safe=False,
    install_requires=['numpy', 'scipy', 'matplotlib', 'astropy', 'mpdaf', 'six'],
    tests_require=['pytest'],
)

from setuptools import setup, find_packages

setup(
    name='origin',
    version='1.0',
    packages=find_packages(),
    package_data={'origin': ['Dico_FWHM_2_12.mat']},
    zip_safe=False,
    install_requires=['numpy', 'scipy', 'matplotlib', 'astropy', 'mpdaf', 'six'],
    tests_require=['pytest'],
)

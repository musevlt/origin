from setuptools import setup, find_packages

setup(
    name='origin',
    version='2.0beta',
    packages=find_packages(),
    package_data={'origin': ['Dico_FWHM_2_12.fits']},
    zip_safe=False,
    install_requires=['numpy', 'scipy', 'matplotlib', 'astropy', 'mpdaf', 'six', 'tqdm'],
    tests_require=['pytest'],
)

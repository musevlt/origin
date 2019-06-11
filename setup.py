import sys

from setuptools import find_packages, setup

if sys.version_info[:2] < (3, 5):
    sys.exit('Origin supports Python 3.5+ only')

setup(
    name='origin',
    use_scm_version=True,
    setup_requires=['setuptools_scm'],
    packages=find_packages(),
    include_package_data=True,
    package_data={'origin': ['Dico_FWHM_2_12.fits']},
    zip_safe=False,
    python_requires='>=3.5',
    install_requires=['numpy', 'scipy', 'matplotlib', 'astropy', 'mpdaf',
                      'tqdm', 'joblib', 'PyYAML', 'photutils>=0.5'],
    tests_require=['pytest'],
)

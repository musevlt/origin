import os
from setuptools import setup, find_packages
from subprocess import check_output

# Read version.py
__version__ = None
__description__ = None
with open('origin/version.py') as f:
    exec(f.read())

# If the version is not stable, we can add a git hash to the __version__
if '.dev' in __version__:
    # Find hash for __githash__ and dev number for __version__ (can't use hash
    # as per PEP440)
    command_hash = 'git rev-list --max-count=1 --abbrev-commit HEAD'
    command_number = 'git rev-list --count HEAD'

    try:
        commit_hash = check_output(command_hash, shell=True)\
            .decode('ascii').strip()
        commit_number = check_output(command_number, shell=True)\
            .decode('ascii').strip()
    except Exception:
        pass
    else:
        # We write the git hash and value so that they gets frozen if installed
        with open(os.path.join('origin', '_githash.py'), 'w') as f:
            f.write("__githash__ = \"{}\"\n".format(commit_hash))
            f.write("__dev_value__ = \"{}\"\n".format(commit_number))

        # We modify __version__ here too for commands such as egg_info
        # __version__ += commit_number

setup(
    name='origin',
    version=__version__,
    description=__description__,
    packages=find_packages(),
    include_package_data=True,
    package_data={'origin': ['Dico_FWHM_2_12.fits']},
    zip_safe=False,
    install_requires=['numpy', 'scipy', 'matplotlib', 'astropy', 'mpdaf',
                      'tqdm', 'joblib', 'PyYAML', 'photutils'],
    tests_require=['pytest'],
)

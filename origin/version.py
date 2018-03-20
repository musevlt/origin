__version__ = '3.0.dev'
# __description__ = ''


def _update_git_version():
    import subprocess
    command_number = 'git rev-list --count HEAD'
    try:
        commit_number = subprocess.check_output(command_number, shell=True)\
            .decode('ascii').strip()
    except Exception:
        pass
    else:
        return commit_number


try:
    if '.dev' in __version__:
        commit_number = _update_git_version()
        if commit_number:
            __version__ += commit_number
        else:
            from ._githash import __dev_value__
            __version__ += __dev_value__
except Exception:
    pass

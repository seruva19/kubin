import platform


def is_windows():
    return platform.system() == "Windows"


def is_linux():
    return platform.system() == "Linux"

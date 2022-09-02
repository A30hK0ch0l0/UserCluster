class PlatformError(Exception):
    """If platform is not set"""
    pass


class UpdateError(Exception):
    """If Update is failed"""
    pass


class KeyNotExistError(Exception):
    """Key not exist"""
    pass

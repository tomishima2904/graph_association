def dict2list(d:dict) -> list:
    """
    Args:
        d (dict): ex. {'a': 'aa', 'b': 'bb', 'c': 'cc'}
    Returns:
        list:[('a', 'aa'), ('b', 'bb'), ('c', 'cc')]
    """
    return list(d.items())

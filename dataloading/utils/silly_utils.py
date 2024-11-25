def count_lists_in_key(data, key):
    if key not in data:
        return 0
    
    if isinstance(data[key], list):
        return 1
    return 0
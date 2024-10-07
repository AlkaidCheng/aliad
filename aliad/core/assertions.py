
def assert_range(name, value, rmin, rmax):
    if (value < rmim) or (value > rmax):
        raise ValueError(f'{name} must be inside the range [{rmin}, {rmax}], but got value = {value}')
import awkward as ak

def get_record_outer_shapes(array):
    nums = ak.num(array).to_list()
    return list(nums.keys()), list(nums.values())
from DataBuilders.fisherman import FishermanDataBuilder
from DataBuilders.stallion import StallionDataBuilder
from DataBuilders.synthetic import SyntheticDataBuilder
from DataBuilders.electricity import ElectrictyDataBuilder


def get_data_helper(dataset_name):
    if dataset_name == "2_fisherman":
        return FishermanDataBuilder()
    elif dataset_name == "synthetic":
        return SyntheticDataBuilder()
    elif dataset_name == "stallion":
        return StallionDataBuilder()
    elif dataset_name == "electricity":
        return ElectrictyDataBuilder()
    else:
        raise ValueError()
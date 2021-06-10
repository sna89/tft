from DataBuilders.fisherman import FishermanDataBuilder
from DataBuilders.stallion import StallionDataBuilder
from DataBuilders.synthetic import SyntheticDataBuilder
from DataBuilders.electricity import ElectricityDataBuilder


def get_data_helper(dataset_name):
    if dataset_name == "2_fisherman":
        return FishermanDataBuilder()
    elif dataset_name == "synthetic":
        return SyntheticDataBuilder()
    elif dataset_name == "stallion":
        return StallionDataBuilder()
    elif dataset_name == "electricity":
        return ElectricityDataBuilder()
    else:
        raise ValueError()
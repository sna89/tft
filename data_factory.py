from DataBuilders.fisherman import FishermanDataBuilder
from DataBuilders.stallion import StallionDataBuilder
from DataBuilders.synthetic import SyntheticDataBuilder
from DataBuilders.electricity import ElectricityDataBuilder


def get_data_builder(config, dataset_name):
    if dataset_name == "2_fisherman":
        return FishermanDataBuilder(config)
    elif dataset_name == "Synthetic":
        return SyntheticDataBuilder(config)
    elif dataset_name == "Stallion":
        return StallionDataBuilder(config)
    elif dataset_name == "Electricity":
        return ElectricityDataBuilder(config)
    else:
        raise ValueError()
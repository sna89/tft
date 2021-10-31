from DataBuilders.fisherman import FishermanDataBuilder
from DataBuilders.stallion import StallionDataBuilder
from DataBuilders.synthetic import SyntheticDataBuilder
from DataBuilders.electricity import ElectricityDataBuilder
from DataBuilders.straus import StrausDataBuilder
from DataBuilders.fisherman2 import Fisherman2DataBuilder


def get_data_builder(config, dataset_name):
    if dataset_name == "Fisherman":
        return FishermanDataBuilder(config)
    elif dataset_name == "Fisherman2":
        return Fisherman2DataBuilder(config)
    elif dataset_name == "Synthetic":
        return SyntheticDataBuilder(config)
    elif dataset_name == "Stallion":
        return StallionDataBuilder(config)
    elif dataset_name == "Electricity":
        return ElectricityDataBuilder(config)
    elif dataset_name == "Straus":
        return StrausDataBuilder(config)
    else:
        raise ValueError()
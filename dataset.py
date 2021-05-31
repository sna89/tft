from constants import DataConst, DataSetRatio
from pytorch_forecasting import TimeSeriesDataSet, GroupNormalizer, NaNLabelEncoder
from pytorch_forecasting.data.examples import get_stallion_data, generate_ar_data
import pandas as pd


def get_data(name="synthetic"):
    if name == "stallion":
        return get_stallion_data()
    else:
        data = generate_ar_data(seasonality=10, timesteps=600, n_series=5, trend=0.5, noise=0.05)
        data["date"] = pd.Timestamp("2020-01-01") + pd.to_timedelta(data.time_idx, "D")
    return data


def create_datasets(data, name="synthetic"):
    training_max_idx = int(data.time_idx.max() * DataSetRatio.TRAIN)
    train_df = data[lambda x: x.time_idx <= training_max_idx]

    test_start_idx = int(data.time_idx.max() * (DataSetRatio.TRAIN + DataSetRatio.VAL)) - (DataConst.ENCODER_LENGTH + DataConst.PREDICTION_LENGTH)
    test_df = data[lambda x: x.time_idx > test_start_idx]

    validation_start_idx = training_max_idx + 1 - (DataConst.ENCODER_LENGTH + DataConst.PREDICTION_LENGTH)
    validation_end_idx = test_start_idx + DataConst.ENCODER_LENGTH + DataConst.PREDICTION_LENGTH
    validation_df = data[lambda x: (x.time_idx > validation_start_idx) & (x.time_idx < validation_end_idx)]

    if name == "stallion":
        train_ts_ds = TimeSeriesDataSet(
            train_df,
            time_idx="time_idx",
            target="volume",
            group_ids=["agency", "sku"],
            min_encoder_length=DataConst.ENCODER_LENGTH,  # keep encoder length long (as it is in the validation set)
            max_encoder_length=DataConst.ENCODER_LENGTH,
            min_prediction_length=DataConst.PREDICTION_LENGTH,
            max_prediction_length=DataConst.PREDICTION_LENGTH,
            static_categoricals=["agency", "sku"],
            static_reals=["avg_population_2017", "avg_yearly_household_income_2017"],
            time_varying_known_categoricals=["special_days", "month"],
            variable_groups={"special_days": DataConst.SPECIAL_DAYS},
            # group of categorical variables can be treated as one variable
            time_varying_known_reals=["time_idx"],
            time_varying_unknown_categoricals=[],
            time_varying_unknown_reals=[
                "volume",
                "industry_volume",
                "soda_volume",
                "log_industry_volume",
                "log_soda_volume",
                "avg_max_temp",
                "price_regular",
                "price_actual",
                "discount",
                "discount_in_percent"
            ],
            target_normalizer=GroupNormalizer(
                groups=["agency", "sku"], transformation="softplus"
            ),  # use softplus and normalize by group
            add_relative_time_idx=True,
            add_target_scales=True,
            add_encoder_length=True,
        )
    else:
        train_ts_ds = TimeSeriesDataSet(
            train_df,
            time_idx="time_idx",
            target="value",
            group_ids=["series"],
            min_encoder_length=DataConst.ENCODER_LENGTH,
            max_encoder_length=DataConst.ENCODER_LENGTH,
            min_prediction_length=DataConst.PREDICTION_LENGTH,
            max_prediction_length=DataConst.PREDICTION_LENGTH,
            time_varying_unknown_reals=["value"],
            time_varying_known_reals=["time_idx"],
            time_varying_known_categoricals=["month", "day"],
            target_normalizer=GroupNormalizer(groups=["series"]),
            add_relative_time_idx=True,
            add_target_scales=True,
            randomize_length=None,
        )

    parameters = train_ts_ds.get_parameters()
    validation_ts_ds = TimeSeriesDataSet.from_parameters(parameters, validation_df)
    test_ts_ds = TimeSeriesDataSet.from_parameters(parameters, test_df)
    return train_ts_ds, validation_ts_ds, test_ts_ds

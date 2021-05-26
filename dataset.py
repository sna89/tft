from constants import DataConst
from pytorch_forecasting import TimeSeriesDataSet, GroupNormalizer


def create_train_val_time_series_datasets(data):
    training_cutoff = data.time_idx.max() - DataConst.PREDICTION_LENGTH

    training_timeseries_ds = TimeSeriesDataSet(
        data[lambda x: x.time_idx <= training_cutoff],
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
    validation_timeseries_ds = TimeSeriesDataSet.from_dataset(training_timeseries_ds, data, predict=True,
                                                              stop_randomization=True)
    return training_timeseries_ds, validation_timeseries_ds

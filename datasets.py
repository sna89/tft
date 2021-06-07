from constants import DataConst, DataSetRatio, SyntheticDataSetParams, Paths
from preprocess import preprocess_synthetic, preprocess_single_df_fisherman
from pytorch_forecasting import TimeSeriesDataSet, GroupNormalizer, NaNLabelEncoder
from pytorch_forecasting.data.examples import get_stallion_data, generate_ar_data
import pandas as pd
import os
import datetime
from datetime import timedelta


def get_2_fisherman_data():
    dfs = []
    max_min_date = datetime.datetime(year=1900, month=1, day=1)
    min_max_date = datetime.datetime(year=2222, month=1, day=1)
    for filename in os.listdir(Paths.FISHERMAN):
        full_file_path = os.path.join(Paths.FISHERMAN, filename)
        df = pd.read_csv(full_file_path, usecols=['Type', 'Value', 'Time'])
        df = df[df.Type == 'internaltemp']
        df = df.drop(columns=['Type'], axis=1)
        df['Time'] = pd.to_datetime(df['Time'])

        min_date_df = df.Time.min()
        if min_date_df > max_min_date:
            max_min_date = min_date_df

        max_date_df = df.Time.max()
        if max_date_df < min_max_date:
            min_max_date = max_date_df

        df['Sensor'] = filename.replace('Sensor ', '').replace('.csv', '')
        dfs.append(df)

    dfs = list(map(lambda dfx: preprocess_single_df_fisherman(dfx,
                                                              max_min_date + timedelta(minutes=10),
                                                              min_max_date - timedelta(minutes=10)
                                                              ),
                   dfs)
               )
    data = pd.concat(dfs, axis=0)
    data.reset_index(inplace=True, drop=True)
    return data


def get_synthetic_data():
    data = generate_ar_data(seasonality=SyntheticDataSetParams.SEASONALITY,
                            timesteps=600,
                            n_series=SyntheticDataSetParams.SERIES,
                            trend=SyntheticDataSetParams.TREND,
                            noise=0.05)
    data["date"] = pd.Timestamp("2020-01-01") + pd.to_timedelta(data.time_idx, "D")
    return data


def define_stallion_ts_ds(train_df):
    stallion_train_ts_ds = TimeSeriesDataSet(
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
    return stallion_train_ts_ds


def define_synthetic_ts_ds(train_df):
    synthetic_train_ts_ds = TimeSeriesDataSet(
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
        # target_normalizer=GroupNormalizer(groups=["series"]),
        add_relative_time_idx=True,
        add_target_scales=True,
        randomize_length=None,
    )
    return synthetic_train_ts_ds


def define_2_fisherman_ts_ds(train_df):
    fisherman_train_ts_ds = TimeSeriesDataSet(
        train_df,
        time_idx="time_idx",
        target="Value",
        group_ids=["Sensor"],
        min_encoder_length=DataConst.ENCODER_LENGTH,  # keep encoder length long (as it is in the validation set)
        max_encoder_length=DataConst.ENCODER_LENGTH,
        min_prediction_length=DataConst.PREDICTION_LENGTH,
        max_prediction_length=DataConst.PREDICTION_LENGTH,
        static_categoricals=["Sensor"],
        static_reals=[],
        time_varying_known_categoricals=["Minute", "Hour", "DayOfMonth", "DayOfWeek"],
        time_varying_known_reals=["time_idx"],
        time_varying_unknown_categoricals=[],
        time_varying_unknown_reals=[
            "Value"
        ],
        # target_normalizer=GroupNormalizer(
        #     groups=["Sensor"]
        # ),
        add_relative_time_idx=True,
        add_target_scales=True,
        add_encoder_length=False
    )
    return fisherman_train_ts_ds


def get_data(name="synthetic"):
    data = pd.DataFrame()
    if name == "stallion":
        data = get_stallion_data()
    elif name == "synthetic":
        data = get_synthetic_data()
        data = preprocess_synthetic(data)
    elif name == "2_fisherman":
        data = get_2_fisherman_data()
    else:
        ValueError('No such dataset')
    return data


def split_dataframe_train_val_test(data):
    training_max_idx = int(data.time_idx.max() * DataSetRatio.TRAIN)
    train_df = data[lambda x: x.time_idx <= training_max_idx]

    test_start_idx = int(data.time_idx.max() * (DataSetRatio.TRAIN + DataSetRatio.VAL)) - (
            DataConst.ENCODER_LENGTH + DataConst.PREDICTION_LENGTH)
    test_df = data[lambda x: x.time_idx > test_start_idx]

    validation_start_idx = training_max_idx + 1 - (DataConst.ENCODER_LENGTH + DataConst.PREDICTION_LENGTH)
    validation_end_idx = test_start_idx + DataConst.ENCODER_LENGTH + DataConst.PREDICTION_LENGTH
    validation_df = data[lambda x: (x.time_idx > validation_start_idx) & (x.time_idx < validation_end_idx)]
    return train_df, validation_df, test_df


def create_datasets(data, name="synthetic"):
    train_df, validation_df, test_df = split_dataframe_train_val_test(data)

    if name == "stallion":
        train_ts_ds = define_stallion_ts_ds(train_df)
    elif name == "synthetic":
        train_ts_ds = define_synthetic_ts_ds(train_df)
    elif name == "2_fisherman":
        train_ts_ds = define_2_fisherman_ts_ds(train_df)
    else:
        raise ValueError

    parameters = train_ts_ds.get_parameters()
    validation_ts_ds = TimeSeriesDataSet.from_parameters(parameters, validation_df)
    test_ts_ds = TimeSeriesDataSet.from_parameters(parameters, test_df)
    return train_ts_ds, validation_ts_ds, test_ts_ds

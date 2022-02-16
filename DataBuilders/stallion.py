from pytorch_forecasting.data.examples import get_stallion_data
from pytorch_forecasting import TimeSeriesDataSet, GroupNormalizer
from Utils.data_utils import add_dt_columns, add_log_column
from DataBuilders.data_builder import DataBuilder


class StallionDataBuilder(DataBuilder):
    SPECIAL_DAYS = [
        "easter_day",
        "good_friday",
        "new_year",
        "christmas",
        "labor_day",
        "independence_day",
        "revolution_day_memorial",
        "regional_games",
        "fifa_u_17_world_cup",
        "football_gold_cup",
        "beer_capital",
        "music_fest",
    ]

    def __init__(self):
        super().__init__()

    @staticmethod
    def build_data():
        data = get_stallion_data()
        return data

    @classmethod
    def get_special_days(cls):
        return cls.SPECIAL_DAYS

    @staticmethod
    def convert_special_days_to_categorical(data):
        special_days = StallionDataBuilder.get_special_days()
        data[special_days] = data[special_days].apply(lambda x: x.map({0: "-", 1: x.name})).astype(
            "category")

    @staticmethod
    def add_time_idx_column(data):
        data['time_idx'] = data.date.dt.year * 12 + data.date.dt.month
        data['time_idx'] -= data.time_idx.min()
        return data

    @staticmethod
    def preprocess(data):
        data.drop(['timeseries'], axis=1, inplace=True)
        StallionDataBuilder.add_time_idx_column(data)
        data = add_dt_columns(data, ['month'])
        add_log_column(data, 'industry_volume')
        add_log_column(data, 'soda_volume')
        return data

    def define_regression_ts_ds(self, train_df):
        special_days = StallionDataBuilder.get_special_days()
        stallion_train_ts_ds = TimeSeriesDataSet(
            train_df,
            time_idx="time_idx",
            target="volume",
            group_ids=["agency", "sku"],
            min_encoder_length=self.enc_length,
            # keep encoder length long (as it is in the validation set)
            max_encoder_length=self.enc_length,
            min_prediction_length=self.prediction_length,
            max_prediction_length=self.prediction_length,
            static_categoricals=["agency", "sku"],
            static_reals=["avg_population_2017", "avg_yearly_household_income_2017"],
            time_varying_known_categoricals=["special_days", "month"],
            variable_groups={"special_days": special_days},
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
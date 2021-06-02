class DataConst:
    PREDICTION_LENGTH = 20
    ENCODER_LENGTH = 100
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


class HyperParameters:
    BATCH_SIZE = 16


class DataSetRatio:
    TRAIN = 0.6
    VAL = 0.2


class DataSetParams:
    SERIES = 1
    SEASONALITY = 30
    TREND = 2


class Paths:
    FISHERMAN = 'FishermanData'
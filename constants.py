class DataConst:
    PREDICTION_LENGTH = 7
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
    BATCH_SIZE = 32


class DataSetRatio:
    TRAIN = 0.6
    VAL = 0.2

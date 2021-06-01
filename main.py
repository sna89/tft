import pandas as pd
from plot import plot_predictions, plot_synthetic_data
from preprocess import preprocess_synthetic
from dataset import create_datasets, get_data
from constants import HyperParameters
from Models.tft import create_trainer, create_tft_model, fit, evaluate, evaluate_base_model, get_fitted_model

pd.set_option('display.max_columns', None)

if __name__ == '__main__':
    data = get_data("synthetic")
    plot_synthetic_data(data)
    preprocess_synthetic(data)
    train, val, test = create_datasets(data)
    train_dataloader = train.to_dataloader(train=True, batch_size=HyperParameters.BATCH_SIZE, num_workers=8)
    val_dataloader = val.to_dataloader(train=False, batch_size=HyperParameters.BATCH_SIZE, num_workers=8)
    test_dataloader = test.to_dataloader(train=False, batch_size=HyperParameters.BATCH_SIZE, num_workers=8)

    print(evaluate_base_model(val_dataloader))
    print(evaluate_base_model(test_dataloader))

    trainer = create_trainer()
    tft_model = create_tft_model(train)
    trainer = fit(trainer, tft_model, train_dataloader, val_dataloader)

    print(evaluate(trainer, val_dataloader))
    print(evaluate(trainer, test_dataloader))

    model = get_fitted_model(trainer)
    plot_predictions(model, test_dataloader)

    # print(evaluate(trainer, val_dataloader))
    # print(evaluate_base_model(val_dataloader))
    # data = data[(data.agency == 'Agency_25') & (data.sku == 'SKU_03')]
    # print(data)
    # plot_volume_by_group(data, 'Agency_25')


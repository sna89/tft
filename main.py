from pytorch_forecasting.data.examples import get_stallion_data
import pandas as pd
from plot import plot_volume_by_group
from preprocess import preprocess
from dataset import create_train_val_time_series_datasets
from constants import HyperParameters
from Models.tft import create_trainer, create_tft_model, fit, evaluate, evaluate_base_model

pd.set_option('display.max_columns', None)

data = get_stallion_data()
preprocess(data)
training, validation = create_train_val_time_series_datasets(data)
train_dataloader = training.to_dataloader(train=True, batch_size=HyperParameters.BATCH_SIZE, num_workers=0)
val_dataloader = validation.to_dataloader(train=False, batch_size=HyperParameters.BATCH_SIZE, num_workers=0)

trainer = create_trainer()
tft_model = create_tft_model(training, train_dataloader, val_dataloader)
trainer = fit(trainer, tft_model, train_dataloader, val_dataloader)
print(evaluate(trainer, val_dataloader))
print(evaluate_base_model(val_dataloader))
# data = data[(data.agency == 'Agency_25') & (data.sku == 'SKU_03')]
# print(data)
# plot_volume_by_group(data, 'Agency_25')


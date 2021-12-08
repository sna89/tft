import plotly.express as px
import os
from CreateSyntheticDataset.generate_data import generate_parameters, create_data_all

VERSIONS = 1
NUM_SERIES = 10
NUM_CORRELATED_LIST = [0, 5, 10]
NUM_SUB_SERIES = 10
TIMESTEPS_SUB_SERIES = 200
noise = 0.25
trend = 1
level = 1

if __name__ == "__main__":
    for version in range(VERSIONS):
        for num_correlated in NUM_CORRELATED_LIST:
            linear_trends_list, levels, seasonality_amp, seasonality_f = generate_parameters(NUM_SERIES,
                                                                                             NUM_SUB_SERIES,
                                                                                             TIMESTEPS_SUB_SERIES,
                                                                                             num_correlated=num_correlated,
                                                                                             level=level)

            data = create_data_all(NUM_SERIES,
                                   NUM_SUB_SERIES,
                                   TIMESTEPS_SUB_SERIES,
                                   linear_trends_list,
                                   seasonality_amp,
                                   seasonality_f,
                                   levels,
                                   noise=noise,
                                   trend=trend)

            data.reset_index(drop=True, inplace=True)

            filename = "SyntheticData_Version_{}_Correlated_{}.<suffix>".format(version, num_correlated)
            data.to_csv(os.path.join(os.getcwd(), filename.replace("<suffix>", "csv")))

            fig = px.line(data, y="Value", x="time_idx", color='Series')
            fig.write_html(os.path.join(os.getcwd(), filename.replace("<suffix>", "html")))

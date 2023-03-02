import os
import datetime
import warnings
import itertools
import matplotlib
import numpy as np
import pandas as pd
from pylab import rcParams
import statsmodels.api as sm
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")
plt.style.use('fivethirtyeight')

matplotlib.rcParams['axes.labelsize'] = 14
matplotlib.rcParams['xtick.labelsize'] = 12
matplotlib.rcParams['ytick.labelsize'] = 12
matplotlib.rcParams['text.color'] = 'k'

errors = pd.DataFrame(
    columns=['File Name', 'ARIMA parameters', 'ARIMA seasonal parameters', 'AIC', 'MSE', 'RMSE'])
output = pd.DataFrame(columns=['type', 'time', 'number', 'range'])

print("IMPORTANT: A csv input file is needed with this format: \n  ID;Number;Time;Date;Day\n")

FILE_PATH = input("Enter csv file full path:")

FILE_NAME = FILE_PATH.split('/', )
FILE_NAME = FILE_NAME[len(FILE_NAME)-1].split('.csv')[0]

alldata = pd.read_csv(FILE_PATH,
                    sep=';',
                    header=None,
                    index_col=False,
                    dtype={0: int, 1: float, 2: str, 3: str, 4: str},
                    )
alldata = alldata.dropna()
#data = alldata[1]

# index created by existing dates

data = pd.DataFrame(columns=['number', 'time', ])
data['number'] = alldata[1]
data['time'] = pd.to_datetime(alldata[3] + ' ' + alldata[2])
data = data.dropna()
data = data[len(data)-100:len(data)]


best_aic = 100000
best_param = 0
best_seasonal = 0

y = data['number'].copy()
y.index = pd.DatetimeIndex(
    freq="m", start=0, periods=len(y))  # data['time']

y.plot(figsize=(15, 6))
plt.show()

rcParams['figure.figsize'] = 18, 8
decomposition = sm.tsa.seasonal_decompose(y, model='additive')
fig = decomposition.plot()
plt.show()


# Time series forecasting with ARIMA
# Find best parameters using grid
p = d = q = range(0, 2)
pdq = list(itertools.product(p, d, q))
seasonal_pdq = [(x[0], x[1], x[2], 12)
                for x in list(itertools.product(p, d, q))]
for param in pdq:
    for param_seasonal in seasonal_pdq:
        try:
            mod = sm.tsa.statespace.SARIMAX(y,
                                            order=param,
                                            seasonal_order=param_seasonal,
                                            enforce_stationarity=False,
                                            enforce_invertibility=False)
            results = mod.fit()
            if results.aic < best_aic:
                best_aic = results.aic
                best_param = param
                best_seasonal = param_seasonal
        except:
            continue

# Fitting the ARIMA model
mod = sm.tsa.statespace.SARIMAX(y,
                                order=best_param,
                                seasonal_order=best_seasonal,
                                enforce_stationarity=False,
                                enforce_invertibility=False)
results = mod.fit()

# Draw results
print(results.summary().tables[1])
results.plot_diagnostics(figsize=(16, 8))
plt.show()

# Validating forecasts
YEAR = round(1970 + (len(y)/12) * 0.8)

pred = results.get_prediction(
    start=pd.to_datetime(str(YEAR)+'-01-31'), dynamic=False)
pred_ci = pred.conf_int()

ax = y['1970':].plot(label='observed')
pred.predicted_mean.plot(
    ax=ax, label='One-step ahead Forecast', alpha=.7, figsize=(14, 7))
ax.fill_between(pred_ci.index,
                pred_ci.iloc[:, 0],
                pred_ci.iloc[:, 1], color='k', alpha=.2)
ax.set_xlabel('Date')
ax.set_ylabel('Numbers')
# set the ylim to bottom, top
plt.ylim(bottom=y.min()-1, top=y.max()+10)
plt.legend()
plt.show()

# Measuring errors
y_forecasted = pred.predicted_mean
y_truth = y[str(YEAR)+'-01-31':]
mse = ((y_forecasted - y_truth) ** 2).mean()

print('The Mean Squared Error of our forecasts is {}'.format(round(mse, 2)))
print('The Root Mean Squared Error of our forecasts is {}'.format(
    round(np.sqrt(mse), 2)))

# Producing and visualizing forecasts

pred_uc = results.get_forecast(steps=20)
pred_ci = pred_uc.conf_int(alpha=0.1)

ax = y.plot(label='observed', figsize=(14, 7))
pred_uc.predicted_mean.plot(ax=ax, label='Forecast')
ax.fill_between(pred_ci.index,
                pred_ci.iloc[:, 0],
                pred_ci.iloc[:, 1], color='k', alpha=.25)
ax.set_xlabel('Date')
ax.set_ylabel('Numbers')
plt.legend()
plt.show()

output['time'] = data['time']
output['number'] = data['number']
output['type'] = 'real'

final_date = data['time'][len(alldata)-1]

for pred_cnt in range(0, 10, 1):
    output.loc[len(output)] = ['predicted', str(
        final_date + pd.DateOffset(minutes=pred_cnt+1)), pred_uc.predicted_mean[pred_cnt],
        '[ ' + str(pred_ci[pred_ci.keys()[0]][pred_cnt]) + ' , ' + str(pred_ci[pred_ci.keys()[1]][pred_cnt]) + ' ]']

errors.loc[len(errors)] = [FILE_NAME, best_param,
                        best_seasonal, best_aic, round(mse, 2), round(np.sqrt(mse), 2)]
print(errors)

with open('output_' + FILE_NAME + '_' + str(datetime.datetime.now()) + '.csv', 'a+') as f:
    output.to_csv(f, header=True)

with open('errors_' + str(datetime.datetime.now()).replace(':', '-') + '.csv', 'a+') as f:
    errors.to_csv(f, header=True)

print("Output is saved in a file named: output_[input file name]_date.csv in program folder.")
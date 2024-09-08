# 同时添加如下代码, 这样每次环境(kernel)启动的时候只要运行下方代码即可: 
# Also add the following code, 
# so that every time the environment (kernel) starts, 
# just run the following code: 
import sys 
sys.path.append('/home/aistudio/external-libraries')
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error, mean_absolute_error
from math import sqrt
import warnings
warnings.filterwarnings("ignore")

# F:\system\flow forecasting\Futures-forecast-PSO-SVM\isp.csv
#读取存放数据集的csv文件
import pandas as pd
data = pd.read_csv(r'F:\system\flow forecasting\Futures-forecast-PSO-SVM\isp.csv',sep=',',usecols=[0,1])

#切割数据和检测数据的缺失值
data_time=data.iloc[:,0]
data = data.iloc[:, 1:]
data = data.fillna(method='ffill')# 用前一个非缺失值去填充该缺失值
print(data.head())

# data = data[['trade_date', 'open', 'close', 'high', 'low']]
data.plot(subplots=True, figsize=(10, 12))
plt.title(f' zhangshang stock attributes from {data_time[0]} to {data_time[len(data_time)-1]}')
plt.show()

#平稳性检验
from statsmodels.tsa.stattools import adfuller as ADF
adf = ADF(data['data'])
if adf[1] > 0.05:# adf[i]表示对data['data']数据进行1阶差分
    print(u'原始序列经检验不平稳，p值为：%s'%(adf[1]))
else:
    print(u'原始序列经检验平稳，p值为：%s'%(adf[1]))

#采用LB统计量的方法进行白噪声检验
from statsmodels.stats.diagnostic import acorr_ljungbox as acorr
p = acorr(data['data'])
if max(p['lb_pvalue'])< 0.05:
    print(u'原始序列非白噪声序列，p值为：%s'%p['lb_pvalue'])
else:
    print(u'原始序列为白噪声序列，p值为：%s'%p['lb_pvalue'])

# 定义绘图函数plotds
def plotds (xt, nlag=30, fig_size=(12,8)):
    if not isinstance(xt, pd.Series): #判断xt是否是pd.Series类型数据，不是则转化为该类型数据
        xt = pd.Series(xt)
        
    plt.figure(figsize=fig_size)
    plt.plot(xt)# 原始数据时序图
    plt.title("Time Series")
    plt.show()
    
    plt.figure(figsize=fig_size)
    layout = (2, 2)
    ax_acf = plt.subplot2grid(layout, (1, 0))
    ax_pacf = plt.subplot2grid(layout, (1, 1))
    plot_acf(xt, lags=nlag, ax=ax_acf)# 自相关图
    plot_pacf(xt, lags=nlag, ax=ax_pacf)# 偏自相关图
    plt.show()
    
    return None

# plotds(data['data'].dropna(), nlag=50)

#定阶
# import statsmodels.tsa.api as smtsa
data_df = data.copy()
aicVal = []
for ari in range(1, 21):
    for maj in range(0,21):
        try:
            arma_obj = ARIMA(data_df.data.tolist(), order=(ari, 0,maj)).fit()
            aicVal.append([ari, maj, arma_obj.aic])
        except Exception as e:
            print(e,'\n',ari,maj)
            
# print(aicVal)
minaicVal=aicVal[0]
for i in aicVal:
    if i[2]<=minaicVal[2]:
        minaicVal=i

import warnings
warnings.filterwarnings("ignore")
data1=data
data = data['data']
data.dropna(inplace=True)

plt.figure(figsize=(12,6))

plt.plot(data, color='blue', label='网络流量数据')
plt.legend()
plt.show()

def draw_acf_pacf(ts, lags=31):
    f = plt.figure(facecolor='white', figsize=(12, 6))
    ax1 = f.add_subplot(121)
    plot_acf(ts, lags=lags, ax=ax1)
    ax2 = f.add_subplot(122)
    plot_pacf(ts, lags=lags, ax=ax2)
    plt.show()

draw_acf_pacf(data)

# 使用 70% 的数据作为训练集，30% 的数据作为测试集
train_size = int(len(data) * 0.75)
train_data = data[:train_size]
test_data = data[train_size:]
# 训练 ARMA 模型
model = ARIMA(train_data, order=(minaicVal[0], 0, minaicVal[1]))
model_fit = model.fit()
# 预测测试集数据
data['pre'] = model_fit.predict()
data['prea'] = model_fit.forecast(steps=len(test_data))


rmse = sqrt(mean_squared_error(test_data, data['prea']))
mae = mean_absolute_error(test_data, data['prea'])

print(f'RMSE: {rmse:.2f}')
print(f'MAE: {mae:.2f}')
print('-----------------------')
print('只考虑五个点')
rmse = sqrt(mean_squared_error(test_data[1:5], data['prea'][1:5]))
mae = mean_absolute_error(test_data[1:5], data['prea'][1:5])

print(f'RMSE: {rmse:.2f}')
print(f'MAE: {mae:.2f}')

plt.figure(figsize=(12, 6))

#plt.plot(train_data, label='训练集')
plt.plot(data1['data'], label='测试集',color='r')
plt.plot(data['pre'], label='预测值', color='b')
plt.plot(data['prea'], label='预测值', color='b')
plt.show()

plt.figure(figsize=(12, 6))
plt.plot(data1['data'], label='测试集',color='r')
plt.plot(data['pre'], label='预测值', color='b',linestyle='--')
#plt.plot(train_data, label='训练集')
rmse = sqrt(mean_squared_error(train_data, data['pre']))
mae = mean_absolute_error(train_data, data['pre'])

print(f'RMSE: {rmse:.2f}')
print(f'MAE: {mae:.2f}')
print(f'使用的模型参数ARMA({minaicVal[0]},{minaicVal[1]})')

plt.show()

# arma_obj_fin = ARIMA(data_df.data.tolist(), order=(minaicVal[0], 0,minaicVal[1])).fit()
# arma_obj_fin.summary()

# #plot the curves
# data_df["ARMA"] = arma_obj_fin.predict()
# plt.figure(figsize=(10,8))
# plt.plot(data_df['data'].iloc[-100:], color='b', label='Actual')
# plt.plot(data_df["ARMA"].iloc[-100:], color='r', linestyle='--', label='ARMA(2,2)_pre')
# plt.xlabel('index')
# plt.ylabel('close price')
# plt.legend(loc='best')
# plt.show()

# fig = arma_obj_fin.predict(len(data_df)-50, len(data_df)+10)

# predict = arma_obj_fin.predict(start=1, end=len(data_df)+10)
# print(predict[-10:])

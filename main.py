import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn import metrics
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error as mae
import warnings
warnings.filterwarnings('ignore')

df = pd.read_csv('train.csv')

parts = df["date"].str.split("-", n = 3, expand = True)
df["year"]= parts[0].astype('int')
df["month"]= parts[1].astype('int')
df["day"]= parts[2].astype('int')

def which_day(year, month, day):
    d = datetime(year,month,day)
    return d.weekday()

df['weekday'] = df.apply(lambda x: which_day(x['year'], x['month'], x['day']),axis=1)
df.drop('date', axis=1, inplace=True)
df['store'].nunique(), df['item'].nunique()
df['store'].nunique(), df['item'].nunique()
df['weekend'] = df['weekday'].apply(lambda x: 1 if x >= 5 else 0)

df = df[df['sales']<140]
features = df.drop(['sales', 'year'], axis=1)
target = df['sales'].values

X_train, X_val, Y_train, Y_val = train_test_split(features, target, test_size = 0.05, random_state=22)

# Нормализация для ускорения #обучения
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)
model = XGBRegressor()
model.fit(X_train, Y_train)
train_pred = model.predict(X_train)
val_pred = model.predict(X_val)

print('Прогноз:', val_pred.head(5))
print('Ошибка на обучающей выборке: ', mae(Y_train, train_preds))
print('Ошибка на тестовой выборке: ', mae(Y_val, val_pred))

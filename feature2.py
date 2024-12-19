print("Duplicate rows:", data.duplicated().sum())
data['date'] = pd.to_datetime(data['date'], format='%Y-%m-%d')
data.isna().sum()
corr = data.corr()
corr['sales'].sort_values(ascending=False)
data = data.sort_values('date')
data[data['sales'].isna()]
y = data[data['sales'].isna()]
data = data.drop(y.index)
data.isna().sum()
X = data
X = X.reset_index(drop=True)
X

def create_sequences(X, y, time_steps=5):
    X_seq, y_seq = [], []
    for i in range(len(X) - time_steps):
        X_seq.append(X[i:(i + time_steps)])
        y_seq.append(y[i + time_steps])
    return np.array(X_seq), np.array(y_seq)


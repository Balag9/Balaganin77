
X['year'] = X['date'].dt.year


year_counts = X['year'].value_counts()

plt.pie(year_counts, labels=year_counts.index, autopct='%1.1f%%')


plt.title('Distribution of Data Values by Year')


plt.show()

num_train = X.shape[0]
num_test = y.shape[0]

plt.bar(['Training Set', 'Test Set'], [num_train, num_test])

plt.title('Number of Data Values in Each Set')
plt.xlabel('Set')
plt.ylabel('Number of Data Values')

plt.show()

X_bilstm['date'] = X_bilstm['date'].apply(lambda x: x.toordinal())

bilstm_model = Sequential([
    Bidirectional(LSTM(units=50, activation='relu'), input_shape=(X_train_bilstm.shape[1], X_train_bilstm.shape[2])),
    Dense(units=1)
])
bilstm_model.compile(optimizer='adam', loss='mse')

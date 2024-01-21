from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

rmse_vals = []
Y = df3['OilPeakRate']
X = df3.drop('OilPeakRate', axis=1)

for i in range(50):
    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.25)
    regress = LinearRegression()
    regress.fit(x_test, y_test)
    predictions = regress.predict(x_test).reshape(-1,1)
    un_rmse = np.sqrt(mean_squared_error(scaler3.inverse_transform(y_test.to_frame()), scaler3.inverse_transform(predictions)))
    rmse_vals.append(un_rmse)
    print(un_rmse)
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import sklearn.datasets
from sklearn import metrics
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor


def load_boston_dataset():
    # Load the Boston housing dataset
    return sklearn.datasets.load_boston()


def load_data():
    # Load the dataset into a Pandas DataFrame
    house_price_dataset = load_boston_dataset()
    house_price_dataframe = pd.DataFrame(house_price_dataset.data, columns=house_price_dataset.feature_names)
    house_price_dataframe['price'] = house_price_dataset.target
    return house_price_dataframe


def visualize_correlation(dataframe):
    # Create a heatmap to visualize correlation
    correlation = dataframe.corr()
    plt.figure(figsize=(10, 10))
    sns.heatmap(correlation, cbar=True, square=True, fmt='.1f', annot=True, annot_kws={'size': 8}, cmap='Blues')
    plt.title("Correlation Heatmap")
    plt.show()


def train_xgboost_model(X_train, Y_train):
    # Train an XGBoost model
    model = XGBRegressor()
    model.fit(X_train, Y_train)
    return model


def evaluate_model(model, X_test, Y_test):
    # Evaluate the model
    test_data_prediction = model.predict(X_test)
    r_squared = metrics.r2_score(Y_test, test_data_prediction)
    mean_absolute_error = metrics.mean_absolute_error(Y_test, test_data_prediction)
    return r_squared, mean_absolute_error


def main():
    house_price_dataframe = load_data()
    print(house_price_dataframe.head())

    visualize_correlation(house_price_dataframe)

    X = house_price_dataframe.drop(['price'], axis=1)
    Y = house_price_dataframe['price']

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=2)

    print("Data shapes:", X.shape, X_train.shape, X_test.shape)

    model = train_xgboost_model(X_train, Y_train)

    r_squared, mean_absolute_error = evaluate_model(model, X_test, Y_test)

    print("R squared error on test data:", r_squared)
    print("Mean absolute error on test data:", mean_absolute_error)

    plt.scatter(Y_test, model.predict(X_test))
    plt.xlabel("Actual Prices")
    plt.ylabel("Predicted Prices")
    plt.title("Actual Prices vs. Predicted Prices")
    plt.show()


if __name__ == "__main__":
    main()

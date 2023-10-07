# boston_house_price_prediction

This Python code performs several tasks related to the Boston Housing dataset, including data loading, data visualization, machine learning model training, and model evaluation. Here is a description of the code's main components:

Import Statements: The code begins by importing necessary libraries, including matplotlib, pandas, seaborn, sklearn.datasets, sklearn.metrics, sklearn.model_selection, and the XGBRegressor class from the xgboost library.

load_boston_dataset Function: This function loads the Boston Housing dataset using the sklearn.datasets.load_boston() function and returns the dataset.

load_data Function: This function calls load_boston_dataset to load the dataset and then creates a Pandas DataFrame from the dataset's data and feature names. It also adds a "price" column to the DataFrame, representing the target variable (housing prices). The DataFrame is returned.

visualize_correlation Function: This function takes a DataFrame as input and generates a heatmap using seaborn to visualize the correlation between the features in the dataset. The heatmap is displayed using matplotlib.

train_xgboost_model Function: This function trains an XGBoost regression model (XGBRegressor) using the provided training data (X_train and Y_train) and returns the trained model.

evaluate_model Function: This function evaluates the performance of a trained model by making predictions on the test data (X_test) and calculating two metrics: the R-squared score (r_squared) and the mean absolute error (mean_absolute_error). These metrics are returned.

main Function: The main function serves as the entry point of the program. It orchestrates the entire workflow:

It loads the data using load_data.
Prints the first few rows of the loaded data.
Calls visualize_correlation to create and display a correlation heatmap.
Splits the data into training and testing sets using train_test_split.
Trains an XGBoost model using train_xgboost_model.
Evaluates the model's performance on the test data using evaluate_model.
Prints the R-squared and mean absolute error scores.
Finally, it creates a scatter plot to visualize the actual prices vs. predicted prices.
if __name__ == "__main__": main(): This code block ensures that the main function is executed when the script is run directly.

In summary, this code loads the Boston Housing dataset, explores its correlation structure through a heatmap, trains an XGBoost regression model, evaluates the model's performance, and visualizes the results. It demonstrates a typical data exploration and regression modeling workflow.

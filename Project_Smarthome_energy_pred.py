# Import necessary libs

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
%matplotlib inline

seed = 79
np.random.seed(seed)
energy = pd.read_csv("./datasets/energydata_complete.csv")

# Dataset exploration
energy = energy.drop(["date","lights","rv1","rv2"], axis=1)
energy.head()
energy.columns
# Dataset characteristics
print("Number of instances in dataset = {}".format(energy.shape[0]))
print("Total number of columns = {}".format(energy.columns.shape[0]))
print("Column wise count of null values:-")
print(energy.isnull().sum())
# Columns for temperature sensors
all_cols = ["Appliances","T1", "T2", "T3", "T4", "T5", "T6", "T7", "T8", "T9", "RH_1", "RH_2", "RH_3", "RH_4", "RH_5", "RH_6", "RH_7", "RH_8", "RH_9","T_out", "Tdewpoint", "RH_out", "Press_mm_hg", "Windspeed", "Visibility"]
temp_cols = ["T1", "T2", "T3", "T4", "T5", "T6", "T7", "T8", "T9"]
hist = ["Appliances","RH_1","RH_2","RH_3","RH_out","T1","T2","T3","T_out"]
# Columns for humidity sensors
rho_cols = ["RH_1", "RH_2", "RH_3", "RH_4", "RH_5", "RH_6", "RH_7", "RH_8", "RH_9"]

# Columns for weather data
weather_cols = ["T_out", "Tdewpoint", "RH_out", "Press_mm_hg", "Windspeed", "Visibility"]

# Target variable column
target = ["Appliances"]
energy[temp_cols].describe()
energy[rho_cols].describe()
energy[weather_cols].describe()
energy[target].describe()
temp_scatter = pd.plotting.scatter_matrix(energy[temp_cols],diagonal="kde",  figsize=(10, 10))
rho_scatter = pd.plotting.scatter_matrix(energy[rho_cols], diagonal="kde", figsize=(10, 10))
all_scatter = pd.plotting.scatter_matrix(energy[all_cols], diagonal="kde", figsize=(16, 16))
weather_scatter = pd.plotting.scatter_matrix(energy[weather_cols], diagonal="kde", figsize=(10, 10))
histograms = energy[hist].hist(figsize=(8, 8), bins=30)
# Histogram for appliances

plt.xlabel("Appliance Energy Consumption in Wh", fontsize="x-large")
plt.ylabel("No. of instances", fontsize="x-large")

energy["Appliances"].hist(figsize=(16, 8), bins=100)
# To generate all pairs for given columns
from itertools import combinations
from scipy.stats import pearsonr
#energy_nod=energy.drop(["date"],axis=1)
for pair in combinations(energy.columns, 2):
    col_1, col_2 = pair
    # Calculate the coefficient and p-value
    corr_coef, p_val = pearsonr(energy[col_1], energy[col_2])
    # Check for high correlation
    if corr_coef > 0.9 or corr_coef < -0.9:
        # Print details for pairs with high correlation
        print("Column pair : {}, {}".format(*pair))
        print("Correlation coefficient : {}".format(corr_coef))
        print("p-value : {}".format(p_val))
from itertools import combinations
from scipy.stats import pearsonr
for pair in combinations(energy[all_cols],2 ):
    col_1, col_2 = pair
    # Calculate the coefficient and p-value
    corr_coef, p_val = pearsonr(energy[col_1], energy[col_2])
    
    print("Column pair : {}, {}".format(*pair))
    print("Correlation coefficient : {}".format(corr_coef))
    print("p-value : {}".format(p_val))
#Split data into Test and Train dataset
from sklearn.model_selection import train_test_split
train,test = train_test_split(energy,test_size = 0.25, random_state=123)
from sklearn.linear_model import LinearRegression
from time import time

# Prepare the data
X_train = train.drop(["Appliances"],axis=1)
y_train = train["Appliances"]

# Initialize and fit the model
#Benchmark model in LM
benchmark_model = LinearRegression()
start = time()
benchmark_model.fit(X_train, y_train)
end = time()

print("Classifier fitted in {:.3f} seconds".format(end-start))


X_test = test.drop(["Appliances"], axis=1)
y_test = test["Appliances"]

# Print scores on both
print("Score on training data : {:.3f}%".format(benchmark_model.score(X_train, y_train) * 100))
print("Score on testing data : {:.3f}%".format(benchmark_model.score(X_test, y_test) * 100))
#Split data into Test and Train dataset
from sklearn.model_selection import train_test_split
train,test = train_test_split(energy,test_size = 0.25, random_state=123)
# Remove correlated features T6 and T9
train = train.drop(["T6", "T9"], axis=1)
test = test.drop(["T6", "T9"], axis=1)
# Import scaler
from sklearn.preprocessing import StandardScaler

# Scales the data to zero mean and unit variance
standard_scaler = StandardScaler()
# Create dummy dataframes to hold the scaled train and test data
train_scaled = pd.DataFrame(columns=train.columns, index=train.index)
test_scaled = pd.DataFrame(columns=test.columns, index=test.index)
# Store the scaled data in new dataframes
train_scaled[train_scaled.columns] = standard_scaler.fit_transform(train)
test_scaled[test_scaled.columns] = standard_scaler.fit_transform(test)
# Prepare training and testing data
X_train = train_scaled.drop("Appliances", axis=1)
y_train = train_scaled["Appliances"]
X_test = test_scaled.drop("Appliances", axis=1)
y_test = test_scaled["Appliances"]
# To calculate Root mean squared error
from sklearn.metrics import mean_squared_error

# Function to fit the regressor and record its metrics
def pipeline(reg, X_train, y_train, X_test, y_test, **kwargs):
    # Dictionary to hold the properties
    reg_props = {}
    
    # Initialize and fit the regressor while recording the time taken for fitting
    regressor = reg(**kwargs)
    start = time()
    regressor.fit(X_train, y_train)
    end = time()
    
    
    # Store the metrics for the regressor
    reg_props["name"] = reg.__name__
    reg_props["train_time"] = end - start
    reg_props["train_score"] = regressor.score(X_train, y_train)
    reg_props["test_score"] = regressor.score(X_test, y_test)
    reg_props["rmse"] = np.sqrt(mean_squared_error(y_test, regressor.predict(X_test)))
    
    return reg_props
# Import the required Regression algorithms
from sklearn.linear_model import Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, ExtraTreesRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import AdaBoostRegressor,BaggingRegressor

# Function to execute each algorithm through the pipeline
def execute_pipeline():
    # Create the list of algorithms
    regressors = [
        Ridge,
        BaggingRegressor,
        RandomForestRegressor,
        GradientBoostingRegressor,
        ExtraTreesRegressor,
        MLPRegressor
    ]
    
    # To store the properties for each regressor
    props = []
 
    for reg in regressors:
        properites = pipeline(reg, X_train, y_train, X_test, y_test, random_state=seed)
        props.append(properites)
        
    return props
# Consolidate the properties into a DataFrame
def get_properties():
    # Obtain the properties after executing the pipeline
    properties = execute_pipeline()
    
    # Extract each individual property of the Regressors
    names = [prop["name"] for prop in properties]
    train_times = [prop["train_time"] for prop in properties]
    train_scores = [prop["train_score"] for prop in properties]
    test_scores = [prop["test_score"] for prop in properties]
    rmse_vals = [prop["rmse"] for prop in properties]
    
    # Create a DataFrame from these properties
    df = pd.DataFrame(index=names, 
                    data = {
                            "Training times": train_times,
                            "Training scores": train_scores,
                            "Testing scores": test_scores,
                            "RMSE": rmse_vals
                      }
                  )
    
    return df

# Obain the properties in a structured DataFrame after executing the pipeline
properties = get_properties()
# Calculate RMSE for the Benchmark model


# Fit the model
start = time()
benchmark_model.fit(X_train, y_train)
end = time()

properties = pd.concat(
    [properties,
    pd.Series(
    {
        "RMSE": np.sqrt(mean_squared_error(y_test, benchmark_model.predict(X_test))),
        "Training scores": benchmark_model.score(X_train, y_train),
        "Testing scores" :benchmark_model.score(X_test, y_test),
        "Training times": end - start,
        "Name": "Linear Regression (Benchmark)"
    }
    ).to_frame().T.set_index(["Name"])]
)

properties
# Plot to compare the training time of algorithms
plt.ylabel("Training time in seconds", fontsize="large")
properties["Training times"].plot(kind="bar", title="Training time of Regressors")
# Plot to compare the performance of the algorithms on both datasets
ax= properties[["Training scores", "Testing scores", "RMSE"]].plot(kind="bar", title="Performance of each Regressor",fontsize="larger", figsize=(8, 5))
ax.set_ylabel("R2 Score/ RMSE", fontsize="larger")
from sklearn.model_selection import GridSearchCV

# Initialize the best performing regressor
clf = ExtraTreesRegressor(random_state=seed)

# Define the parameter subset
param_grid = {   "n_estimators": [100],
    "max_features": ["auto", "sqrt", "log2"],
    "max_depth": [None, 10, 50, 100]
}

# Use Randomized search to try 20 subsets from parameter space with 5-fold cross validation
grid_search = GridSearchCV(clf, param_grid, n_jobs=1, cv=10)
grid_search.fit(X_train, y_train)
# Display best params
print("Parameters of best Regressor : {}".format(grid_search.best_params_))
best_model = grid_search.best_estimator_

# Display metrics on training and test set
print("R2 score on Training set = {:.3f}".format(best_model.score(X_train, y_train)))
print("RMSE on Training set = {:.3f}".format(np.sqrt(mean_squared_error(y_train, best_model.predict(X_train)))))
print("R2 score on Testing set = {:.3f}".format(best_model.score(X_test, y_test)))
print("RMSE on Testing set = {:.3f}".format(np.sqrt(mean_squared_error(y_test, best_model.predict(X_test)))))
# Find the index of most and least important feature and display that column
print("Most important feature = {}".format(X_train.columns[np.argmax(best_model.feature_importances_)]))
print("Least important feature = {}".format(X_train.columns[np.argmin(best_model.feature_importances_)]))

# Get the indices based on feature importance in ascending order 
feature_indices = np.argsort(best_model.feature_importances_)

print("\nTop 5 most important features:-")
# Reverse the array to get important features at the beginning
for index in feature_indices[::-1][:5]:
    print(X_train.columns[index])
    
print("\nTop 5 least important features:-")
for index in feature_indices[:5]:
    print(X_train.columns[index])

# Plot feature importance

fi = pd.DataFrame(index=X_train.columns[feature_indices], data=np.sort(best_model.feature_importances_))

ax = fi.plot(kind="bar", title="Feature Importances of Extra Tree Regressor",fontsize="larger", figsize=(8, 5))
ax.set_ylabel("Values", fontsize="larger")
ax.legend_.remove()

from sklearn.model_selection import GridSearchCV

# Initialize the best performing regressor
clf = RandomForestRegressor(random_state=seed)

# Define the parameter subset
param_grid = {   "n_estimators": [10],
    "max_features": ["auto", "sqrt", "log2"],
    "max_depth": [None, 10, 50, 100]
}

# Use Randomized search to try 20 subsets from parameter space with 5-fold cross validation
grid_search = GridSearchCV(clf, param_grid, n_jobs=1, cv=10)
grid_search.fit(X_train, y_train)

best_model = grid_search.best_estimator_

# Display metrics on training and test set
print("R2 score on Training set = {:.3f}".format(best_model.score(X_train, y_train)))
print("RMSE on Training set = {:.3f}".format(np.sqrt(mean_squared_error(y_train, best_model.predict(X_train)))))
print("R2 score on Testing set = {:.3f}".format(best_model.score(X_test, y_test)))
print("RMSE on Testing set = {:.3f}".format(np.sqrt(mean_squared_error(y_test, best_model.predict(X_test)))))

# Find the index of most and least important feature and display that column
print("Most important feature = {}".format(X_train.columns[np.argmax(best_model.feature_importances_)]))
print("Least important feature = {}".format(X_train.columns[np.argmin(best_model.feature_importances_)]))

# Get the indices based on feature importance in ascending order 
feature_indices = np.argsort(best_model.feature_importances_)

print("\nTop 5 most important features:-")
# Reverse the array to get important features at the beginning
for index in feature_indices[::-1][:5]:
    print(X_train.columns[index])
    
print("\nTop 5 least important features:-")
for index in feature_indices[:5]:
    print(X_train.columns[index])

# Plot feature importance

fi = pd.DataFrame(index=X_train.columns[feature_indices], data=np.sort(best_model.feature_importances_))

ax = fi.plot(kind="bar", title="Feature Importances Random Forest Regressor",fontsize="large", figsize=(8, 5))
ax.set_ylabel("Values", fontsize="large")
ax.legend_.remove()
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler,PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split


file_name = 'housing.csv'
df = pd.read_csv(filepath_or_buffer=file_name)
print(df.head())

# Question 1
# Display the data types of each column using the function dtypes. Take a screenshot of your code and output. 
# You will need to submit the screenshot for the final project.
for column in df.columns:
    print(column," : ",df[column].dtype)



# Module 2: Data Wrangling
# Question 2
# Drop the columns "id" and "Unnamed: 0" from axis 1 using the method drop(), then use the method describe() 
# to obtain a statistical summary of the data. Make sure the inplace parameter is set to True. 
# Take a screenshot of your code and output. You will need to submit the screenshot for the final project.
df.drop(['id', 'Unnamed: 0'], axis=1, inplace=True)
print(df.describe())

print("number of NaN values for the column bedrooms :", df['bedrooms'].isnull().sum())
print("number of NaN values for the column bathrooms :", df['bathrooms'].isnull().sum())


mean=df['bedrooms'].mean(axis=0)
df['bedrooms'].replace(np.NaN,mean, inplace=True)

mean=df['bathrooms'].mean(axis=0)
df['bathrooms'].replace(np.NaN,mean, inplace=True)


# Module 3: Exploratory Data Analysis
# Question 3 Use the method value_counts to count the number of houses with unique floor values,
# use the method .to_frame() to convert it to a data frame. Take a screenshot of your code and output. 
# You will need to submit the screenshot for the final project.
df_floor = df['floors'].value_counts().to_frame()
print(df_floor)


# Question 4 Use the function boxplot in the seaborn library to determine whether houses with a waterfront view or 
# without a waterfront view have more price outliers. Take a screenshot of your code and boxplot. 
# You will need to submit the screenshot for the final project.
sns.boxplot(data=df, x='waterfront', y='price')
plt.show()

# Question 5¶ 
# Use the function regplot in the seaborn library to determine if the feature sqft_above is negatively or positively correlated with price. 
# Take a screenshot of your code and scatterplot. You will need to submit the screenshot for the final project.
sns.regplot(data=df, x='sqft_above', y='price')
plt.show()

# print(df.corr()['price'].sort_values())


# Module 4: Model Development
# We can Fit a linear regression model using the longitude feature 'long' and caculate the R^2.
lm = LinearRegression()
X = df[['long']]
Y = df['price']

lm.fit(X,Y)
y_pred = lm.predict(X)
print(f'R-Squared Value: {lm.score(X, Y)}')



# Question 6 Fit a linear regression model to predict the 'price' using the feature 'sqft_living' 
# then calculate the R^2. Take a screenshot of your code and the value of the R^2. 
# You will need to submit it for the final project.
lm = LinearRegression()
X = df[['sqft_living']]
Y = df['price']

lm.fit(X, Y)
Y_pred = lm.predict(X)
print(f'R-Squared Value: {lm.score(X, Y)}')


# Question 7¶ Fit a linear regression model to predict the 'price' using the list of features:
# # Then calculate the R^2. Take a screenshot of your code and the value of the R^2. You will need to submit it for the final project.
lm = LinearRegression()
features =["floors", "waterfront","lat" ,"bedrooms" ,"sqft_basement" ,"view" ,"bathrooms",
           "sqft_living15","sqft_above","grade","sqft_living"]

X = df[features]
Y = df['price']
lm.fit(X, Y)
Y_pred = lm.predict(X)
print(f'R-Squared Value: {lm.score(X, Y)}')  
    

# Question 8 Use the list to create a pipeline object to predict the 'price', fit the object using the features 
# in the list features, and calculate the R^2. Take a screenshot of your code and the value of the R^2. You will need to submit it for the final project
features =df[["floors", "waterfront","lat" ,"bedrooms" ,"sqft_basement" ,"view" ,"bathrooms","sqft_living15","sqft_above","grade","sqft_living"]]
Y = df['price']

Input = [('scale', StandardScaler()), ('polynomial', PolynomialFeatures(include_bias=False)), ('model', LinearRegression())]
pipe = Pipeline(Input)
pipe.fit(features, Y)

y_pred = pipe.predict(features)
print(f'R Squared Value: {pipe.score(features, Y)}')


# Module 5: Model Evaluation and Refinement
features =["floors", "waterfront","lat" ,"bedrooms" ,"sqft_basement" ,"view" ,"bathrooms","sqft_living15","sqft_above","grade","sqft_living"]    
X = df[features]
Y = df['price']

x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.15, random_state=1)
print("number of test samples:", x_test.shape[0])
print("number of training samples:",x_train.shape[0])


# Question 9 
# Create and fit a Ridge regression object using the training data, set the regularization parameter to 0.1, 
# and calculate the R^2 using the test data. Take a screenshot of your code and the value of the R^2. 
# You will need to submit it for the final project.
from sklearn.linear_model import Ridge

RidgeModel = Ridge(alpha=0.1)
RidgeModel.fit(x_train, y_train)
print(f'R^2 Value: {RidgeModel.score(x_test, y_test)}')


# Question 10 Perform a second order polynomial transform on both the training data and testing data.
# Create and fit a Ridge regression object using the training data, set the regularisation parameter to 0.1, 
# and calculate the R^2 utilising the test data provided. Take a screenshot of your code and the R^2.
pf = PolynomialFeatures(degree=2)
x_train_pf = pf.fit_transform(x_train)
x_test_pf = pf.fit_transform(x_test)
rig = Ridge(alpha=0.1)
rig.fit(x_train_pf,y_train)
print(f'R^2 Value: {rig.score(x_test_pf,y_test)}')

#!/usr/bin/env python
# coding: utf-8

# # Introduction

# The purpose of this project is to create both a neural network and a logistic regression of an attrition dataset. After creating both models we will then compare both of them to see which one is better.  The data for this particular project will be derived from Kaggle.  
# https://www.kaggle.com/datasets/rishikeshkonapure/hr-analytics-prediction

# # Purpose

# Employee attrition is a common problem many companies today face. While the data for this particular project cannot represent scenarios or variables that every company can face , it does provide a general idea of what may be potential problems. With that being said our goal for this project will be to use the datasets independent variables to help predict attrition. Note, that in this project the target variable will be **‘attrition’**  which is a binary variable of ‘yes’ or ‘no’. Yes, in the fact that the employee quit/ left their job and no in the fact that they did not.  

# # Importing the data and packages 

# In[1]:


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.utils import to_categorical
import tensorflow as tf


# In[2]:


# Load the data
data = pd.read_csv('WA_Fn-UseC_-HR-Employee-Attrition.csv') 


# # Data Exploration

# By running the code data.info() we can both check the size of our data and check if null values exist. The results from running this code tell us data we have 1,470 observations and 35 variables. Note that we have both numerical and categorical variables. Similarly, note that our dataset does not have existing null values.

# In[3]:


data.info()


# In[4]:


#Lets look at the first 5 observations
data.head()


# # Data Cleaning

# Based on data exploration we have categorical variables in our dataset. Therefor we will check the unique number of categories in each variable before creating dummy variables.

# In[5]:


#lets check the number of unique categories in our categorical variables
for column in data.columns:
    if data[column].dtype == 'object':  # if the column is categorical
        num_unique_categories = data[column].nunique()
        print(f'{column}: {num_unique_categories} unique categories')


# ---

# The number of unique categories for each categorical variable is low , therefor we can proceed into making dummy variables. Note that we will first start by hardcoding our target variable **attrition**. The reason behind this is because when we run the code *data = pd.get_dummies(data, drop_first=True)* pandas has a hard time interpreting the 'attrition' variable and drops it from the database.

# In[6]:


data['Attrition'] = data['Attrition'].map({'No': 0, 'Yes': 1})
data = pd.get_dummies(data, drop_first=True)


# ---

# Let us now double check the dummy variables that were created. The data type for categorical variables is now uint8 instead of object. Similarly, note that hardcoding attrition changes its data type to int64.

# In[7]:


#checks the datatypes
data.dtypes


# ___

# Before we can move on to feature scaling we must first convert **Attrition** to a non-numeric data type. If we don’t, attrition will be featured scaled along all the other numeric variables in the dataset. Therefor its best to change Attrition to the same data type as all the other dummy variables (uint8).

# In[8]:


# Convert the data type of Attrition to uint8
data['Attrition'] = data['Attrition'].astype('uint8')


# ---

# # Feature Scaling : Standardization

# If you recall in data exploration, when we used the code '*data.head()*' we had variables like Daily rain and Employee count. Daily rain had values ranging up to a thousand while employee count had values of one. This tells us that our dataset needs to be featured scaled to put variables into a similar scale. For this example, I will use standardization which is a feature scaling method that will scale the predicter value data to have a mean of 0 and a standard deviation of 1.
# 
# Note that only the numeric variables should be scaled. Therefor, were going to isolate the numeric variables and scaled them in a new function called **data_scaled**.

# In[9]:


# Create a scaler object
scaler = StandardScaler()

# Select only the numeric columns
numeric_columns = data.select_dtypes(include=['int64', 'float64'])


# Fit the scaler and transform the data
data_scaled = scaler.fit_transform(numeric_columns)

# Convert back to a dataframe
data_scaled = pd.DataFrame(data_scaled, columns=numeric_columns.columns)


# ---

# Simirlaly were going to also isolate the dummy variables we created and stored them in a new function called **dummy_columns**.

# In[10]:


# Get the dummy variables from the original DataFrame
dummy_columns = data.select_dtypes(include=['uint8'])


# ---

# Now that the numeric variables are scaled in **data_scaled** , we need to merger them together with **dummy_columns** to get an updated dataset that has both featured scaled numeric numbers and dummy variables. Note, that it is important to reset the indices before concatenating, because if we don’t pandas can create errors in the merger.

# In[11]:


# Reset the indices of both DataFrames
data_scaled = data_scaled.reset_index(drop=True)
dummy_columns = dummy_columns.reset_index(drop=True)

# Combine the scaled DataFrame and the dummy variables
data_scaled = pd.concat([data_scaled, dummy_columns], axis=1)


# In[12]:


data_scaled


# # Data Splitting

# Now that our data is scaled and has dummy variables lets split the data into training and testing sets. I first start by creating a new variable called **features** which eliminates the target variable attrition from the rest of the predicter variables. Then I create a variable called **target** which will only include data from the attrition variable. By creating two variables called **features** and **target** I will able to isolate the predicter values from the attrition variable, forming two distinct training(**X_train** and **y_train**) and testing sets (**X_test** and **y_test**). 

# In[13]:


#Splits our data into two variables features(predicter variables) and  target(attrition)
features = data_scaled.drop('Attrition', axis=1)
target = data_scaled['Attrition']

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)


# # Neural Network

# Now we can start building the neural network . For this example I choose to go with three layers and start with 32 units. The ReLU function is preferred in neural networks because it is computationally efficient, easy to implement, and has been shown to work well in similar neural networks.

# In[14]:


# Build the neural network
model = Sequential()
model.add(Dense(32, input_dim=X_train.shape[1], activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(1))


# ---

# Compiling the neural network is next , for this I used 'adam' a very popular optimizer that deals really well with noisy data. Similarly, BinerayCrossentropy is commonly used for binary classification problems like ours. Logits=True argument means that the model's final layer has not been passed through an activation function, and the model is outputting the direct logits of the last layer. This is a common setup when using binary cross-entropy, as it makes the model more numerically stable. Lastly, Accuracy provides a simple measure of the proportion of correctly predicted instances, we will use this measure to compare this models accuracy againts another one.

# In[15]:


# Compile the model
model.compile(optimizer='adam', loss=tf.keras.losses.BinaryCrossentropy(from_logits=True), metrics=['accuracy'])


# ---

# Now we can start training our model with our training data . For this we used 10 epochs and a batch size of 10. 

# In[16]:


# Train the model
model.fit(X_train, y_train, epochs=10, batch_size=10)


# ---

# Afterwards, we can call on the Loss function using the same metrics we used on the compiler.

# In[17]:


#Loss Function
model.compile(optimizer='adam',
              loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
              metrics=['accuracy'])


# ---

# This is where everything gets put together, in summary this code is training the model on the X_train data and y_train labels for 10 epochs, evaluating it against the X_test data and y_test labels after each epoch, and storing the history of this process in the function **history** .

# In[18]:


# where the training of the neural network actually happens
history = model.fit(X_train, y_train, epochs=10, validation_data=(X_test, y_test))


# #### Results

# The neural network is now complete, lets test the accuracy of the model's predictions by comparing them to the true labels of the test set. 
# 
# The results from running the code below indicate that the Neural Network accuracy score was 0.884 which meant that the model correctly predicted the attrition variable ( 0 or 1) about 88.4% of the instances in the test set.

# In[19]:


# Calculate the accuracy of the Neural Network model on the test set
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
nn_predictions = (model.predict(X_test) > 0.5).astype(int)
nn_accuracy = accuracy_score(y_test, nn_predictions)

# Print the accuracy
print('Neural Network accuracy: ', nn_accuracy)


# # Testing the Neural Network

# Let us now test the finished neural network with fictitious data to see if we get an attrition value of (0 or 1). For this I will create a new dataframe called **new_data** which will have fictitious data that is both featured scaled and has applied dummy variables.
# 
# The result from running the code below indicates a value of 1 , which means that the employee given the data  traits we entered will face attrition. Note, that running the code below can result in a different answer every time , as the model is 88% accurate with a 12% chance of making errors.

# In[21]:


new_data = pd.DataFrame({
    'Age': [0.446350],
    'DailyRate': [0.742527],
    'DistanceFromHome': [-1.010909],
    'Education': [-0.891688],
    'EmployeeCount': [0.000000],
    'EmployeeNumber': [-1.701283],
    'EnvironmentSatisfaction': [-0.660531],
    'HourlyRate': [1.383138],
    'JobInvolvement': [0.379672],
    'JobLevel': [-0.057788],
    'JobSatisfaction': [1.153254],
    'MonthlyIncome': [-0.108350],
    'MonthlyRate': [0.726020],
    'NumCompaniesWorked': [2.125136],
    'PercentSalaryHike': [-1.150554],
    'PerformanceRating': [-0.426230],
    'RelationshipSatisfaction': [-1.584178],
    'StandardHours': [0.000000],
    'StockOptionLevel': [-0.932014],
    'TotalWorkingYears': [-0.421642],
    'TrainingTimesLastYear': [-2.171982],
    'WorkLifeBalance': [-2.493820],
    'YearsAtCompany': [-0.164613],
    'YearsInCurrentRole': [-0.063296],
    'YearsSinceLastPromotion': [-0.679146],
    'YearsWithCurrManager': [0.245834],
    'BusinessTravel_Travel_Frequently': [0.000000],
    'BusinessTravel_Travel_Rarely': [1.000000],
    'Department_Research & Development': [0.000000],
    'Department_Sales': [1.000000],
    'EducationField_Life Sciences': [1.000000],
    'EducationField_Marketing': [0.000000],
    'EducationField_Medical': [0.000000],
    'EducationField_Other': [0.000000],
    'EducationField_Technical Degree': [0.000000],
    'Gender_Male': [0.000000],
    'JobRole_Human Resources': [0.000000],
    'JobRole_Laboratory Technician': [0.000000],
    'JobRole_Manager': [0.000000],
    'JobRole_Manufacturing Director': [0.000000],
    'JobRole_Research Director': [0.000000],
    'JobRole_Research Scientist': [0.000000],
    'JobRole_Sales Executive': [1.000000],
    'JobRole_Sales Representative': [0.000000],
    'MaritalStatus_Married': [0.000000],
    'MaritalStatus_Single': [1.000000],
    'OverTime_Yes': [1.000000]
})

# Keep a copy of the original DataFrame
X_train_df = pd.DataFrame(X_train, columns=features.columns)

# Ensure the new data has the same features as the training set
missing_cols = set(X_train_df.columns) - set(new_data.columns)
for c in missing_cols:
    new_data[c] = 0
new_data = new_data[X_train_df.columns]

# Make a prediction
prediction = model.predict(new_data)

# The output is a probability, so you might want to convert it to a class label
prediction_label = (prediction > 0.5).astype(int)

print(prediction_label)


# # Logistic Regression Model

# Using the same data split we used in the neural network (**X_train**,**y_train**,**X_test**, and **y_test**) , lets now create a logistic regression model and compare its accuracy with the neural network.
# 
# The results from running the code below, convey that the logistic regression model is 89.8% accurate. This means that this model is slightly more accurate than the neural network model.

# In[23]:


from sklearn.linear_model import LogisticRegression


# Create a Logistic Regression model
logreg = LogisticRegression(max_iter=1000)

# Train the model
logreg.fit(X_train, y_train)

# Make predictions on the test set
logreg_predictions = logreg.predict(X_test)

# Calculate the accuracy of the Logistic Regression model
logreg_accuracy = accuracy_score(y_test, logreg_predictions)

# Print the accuracy
print('Logistic Regression accuracy: ', logreg_accuracy)


# ---

# # Comparing the models

# Based on these results, the logistic regression model performed slightly better on the test set than the neural network model.
# 
# However, keep in mind that this doesn't necessarily mean that logistic regression is a better choice for this particular problem. The performance of a model can depend on many factors, including the architecture of the model (for neural networks), the hyperparameters, and the way the data is preprocessed.
# 
# It's also important to note that accuracy is just one measure of model performance. Depending on the problem, other metrics like precision, recall, or the area under the ROC curve might be more appropriate.
# 
# Finally, remember that these results here are based on a single split of the data into a training set(80%) and a test set(20%). To get a more reliable estimate of model performance, we could have also used cross-validation, which involved training and testing the models on different splits of the data.

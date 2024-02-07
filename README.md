### iPhone CO2 Emissions Prediction

## Multiple Linear Regression Overview (iPhone_linReg.py)

This project focuses on predicting the CO2 emissions of various iPhone models using a machine learning approach. We employ a Linear Regression model to assess how different features influence CO2 emissions. The dataset iPhone.csv is utilized for this purpose, comprising various iPhone models and their respective specifications.

## Data Processing

We start by checking for missing values in the dataset. It was found that there are no missing values, which simplified the preprocessing stage. To incorporate categorical data (i.e., the iPhone model names), we apply one-hot encoding, transforming the NAME column into multiple binary columns, each representing a different iPhone model.

## Model Training and Evaluation

A Linear Regression model is applied for this task. We evaluate the model's performance using 10-fold cross-validation and calculate the Root Mean Squared Error (RMSE) to assess prediction accuracy. After training, we extract the feature importance to understand which variables significantly influence CO2 emissions.

## Results

The average RMSE across all cross-validation folds is approximately 3.45. The top 5 important models include iPhone 12, iPhone 8, iPhone 13, iPhone 13 mini, and another variant of iPhone 12, indicating the significant impact of the model name on CO2 emissions predictions.

Removing the NAME column and only using numerical features resulted in an average RMSE of approximately 4.07. The top 5 important features based on their significance are Display Size, GPU Cores, Memory, Front Camera Megapixels, and Main Camera Megapixels.

## Impact of Numerical Features on CO2E

DISPLAY (inch): Each additional inch in display size increases CO2 emissions by approximately 19.34 units. This indicates that larger displays are associated with higher CO2 emissions.

GPU (cores): Adding one more core to the GPU increases CO2 emissions by around 9.90 units.

MEMORY (gb): Each additional gigabyte of memory is associated with an increase of about 8.33 units in CO2 emissions.

FRONT CAMERA (mp): Increasing the megapixel count of the front camera by one megapixel leads to an increase of approximately 3.36 units in CO2 emissions.

MAIN CAMERA (mp): Each additional megapixel in the main camera increases CO2 emissions by about 1.35 units.


## Simple Linear Regression Overview (iPhone_simple_linReg.py)

This part of the project aims to quantify the impact of each iPhone features on their CO2 emissions. By employing a Linear Regression model, we isolate the effect of individual features such as display size, storage capacity, and camera specifications on CO2 emissions. The analysis is also based on data from the iPhone.csv file.

## Methodology

The dataset is preprocessed to convert the NAME column into a set of binary variables through one-hot encoding, once again. This allows us to treat each iPhone model as a separate feature in the analysis. For each feature outside the NAME category, we train a Linear Regression model to predict CO2 emissions based on that single feature, while controlling for model variations through one-hot encoding.

## Key Findings

Display Size (inch): Each additional inch in display size is associated with an increase of approximately 11.04 units in CO2 emissions.

Storage (gb): Each additional gigabyte of storage results in a marginal increase of about 0.052 units in CO2 emissions.

Memory (gb): An increase of one gigabyte in memory capacity leads to a rise of approximately 3.27 units in CO2 emissions.

GPU (cores): Each additional GPU core is linked to an increase of about 6.56 units in CO2 emissions.

Front Camera (mp): Each additional megapixel in the front camera specification contributes to an increase of approximately 4.04 units in CO2 emissions.

Number of Cameras (Back): Each additional back camera is significantly associated with a rise of about 13.84 units in CO2 emissions, marking it as the feature with the highest impact on CO2 emissions among those studied.

In this Kaggle competition, we challenged to use reservation and visitation data to predict the total number of visitors to a restaurant for future dates. 

csv2sqlite.py: This file transform original csv type data files into sqlite3 database, which is easy to check data. 

WeatherDataAddressing.py: This file preaddressed external weather data from online.

Forecasting.py: This file is the main file for feature engineering and forecasting. We firstly adopted the time series methods with Facebook Prophet package, whose results were not that good. We annotated this part and start with regression methods.

We constructed a lots of statistical quantities (such as mean, variance and median) of historical visitors for different periods (such as week, month and seasons). We also treat region and type of restaurants as features. 

Finally we have tried normal Gradient Boosting Regression Tree, XGBoost regression and K-nearest neighbor regression models. We found the XGBoost gave rise to best results. After getting final results, we adopted a post addressing based on reservation number. 
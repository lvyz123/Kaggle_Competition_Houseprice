import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
train_data = pd.read_csv('train.csv')
test_data = pd.read_csv('test.csv')
total_feature_set = list(train_data.columns.values)
total_feature_set.remove('Id')
total_feature_set.remove('SalePrice')
le = LabelEncoder()
for column_name in total_feature_set:
    if train_data[column_name].dtype == 'object':
        le.fit(list(train_data[column_name]))
        train_data[column_name] = le.transform(list(train_data[column_name]))
imp = SimpleImputer(missing_values='NA', strategy='mean')
print(np.all(np.isfinite(train_data)))
imp_cleaned = imp.fit_transform(train_data)
train_X = imp_cleaned
train_y = train_data['SalePrice']
test_X = test_data[total_feature_set]
prediction_model = RandomForestRegressor(random_state=1)
prediction_model.fit(train_X, train_y)
prediction_y = prediction_model.predict(test_X)
prediction_result = pd.DataFrame({'Id': test_data['Id'], 'SalePrice': prediction_y})
prediction_result.to_csv('prediction_submission.csv', index=False)

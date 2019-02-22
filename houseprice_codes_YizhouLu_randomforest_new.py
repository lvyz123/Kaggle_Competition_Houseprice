import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor

train_data = pd.read_csv('train.csv')
test_data = pd.read_csv('test.csv')
train_test_data = pd.concat([train_data,test_data])

total_feature_set = list(train_data.columns.values)
total_feature_set.remove('Id')
total_feature_set.remove('SalePrice')

transformed_data = pd.DataFrame()
for column_name in total_feature_set:
    if train_test_data[column_name].dtype == 'object':
        transformed_data = pd.concat([transformed_data,pd.get_dummies(train_test_data[column_name], prefix=column_name)],axis=1)
    else:
        transformed_data = pd.concat([transformed_data,train_test_data[column_name]],axis=1)
        transformed_data[column_name] = transformed_data[column_name].replace('NA','')
        transformed_data[column_name] = transformed_data[column_name].fillna(transformed_data[column_name].mean())

train_X = transformed_data[:1460]
train_y = train_data['SalePrice']
test_X = transformed_data[1460:]
prediction_model = RandomForestRegressor(random_state=1)
prediction_model.fit(train_X, train_y)

prediction_y = prediction_model.predict(test_X)
prediction_result = pd.DataFrame({'Id': test_data['Id'], 'SalePrice': prediction_y})
prediction_result.to_csv('prediction_submission.csv', index=False)

import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

def learning_curve_plot(model,train_X,train_y,val_X,val_y):
    train_errors = [];val_errors = []
    for m in range(5,len(train_y)):
        model.fit(train_X[:m],train_y[:m])
        predict_train_y = model.predict(train_X[:m])
        predict_val_y = model.predict(val_X)
        rmse_train = mean_squared_error(predict_train_y,train_y[:m])
        rmse_val = mean_squared_error(predict_val_y,val_y)
        train_errors.append(rmse_train);val_errors.append(rmse_val)
    plt.plot(np.sqrt(train_errors),'r-+',linewidth=2,label='Train')
    plt.plot(np.sqrt(val_errors),'b-',linewidth=3,label='Val')
    plt.xlabel('Number of Samples')
    plt.ylabel('RMSE')

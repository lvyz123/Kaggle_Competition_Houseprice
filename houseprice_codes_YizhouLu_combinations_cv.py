# =============
# Module Import
# =============
import sys
import pandas as pd
import numpy as np
import learning_curve as lc
from sklearn.linear_model import ElasticNet
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import DotProduct, RBF, Matern, WhiteKernel, CompoundKernel, PairwiseKernel
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import GridSearchCV, cross_val_score, KFold

# ==========
# Data Input
# ==========
train_data = pd.read_csv('train.csv',index_col='Id')
test_data = pd.read_csv('test.csv',index_col='Id')
train_test_data = pd.concat([train_data,test_data],axis=0)

# ====================
# Mapping Dictionaries
# ====================
# A Map of BsmtFinType Scores
BsmtFinType_Dict = {
                    'GLQ': 6,
                    'ALQ': 5,
                    'BLQ': 4,
                    'Rec': 3,
                    'LwQ': 2,
                    'Unf': 1,
                    'NA': 0,
                    }

# A Map of Functional Scores
Functional_Dict = {
                    'Typ': 7,
                    'Min1': 6,
                    'Min2': 5,
                    'Mod': 4,
                    'Maj1': 3,
                    'Maj2': 2,
                    'Sev': 1,
                    'Sal': 0,
                    'NA': 0,
                    }

# A Map of Quality Scores
QuScore_Dict = {
                    'Ex': 5,
                    'Gd': 4,
                    'TA': 3,
                    'Fa': 2,
                    'Po': 1,
                    'NA': 0,
                    }

# A Map of Finishing Scores
FinishScore_Dict = {
                    'Fin': 3,
                    'RFn': 2,
                    'Unf': 1,
                    'NA': 0,
                    }

# A Map of Pave Scores
PaveScore_Dict = {
                    'Y': 2,
                    'P': 1,
                    'N': 0,
                    }

# A Map of Fence Scores
FenceScore_Dict = {
                    'GdPrv': 4,
                    'MnPrv': 3,
                    'GdWo': 2,
                    'MnWw': 1,
                    'NA': 0,
                    }

# =============
# Data Cleaning
# =============
train_test_data.fillna('NA',inplace=True)

# Type of Dwelling
sale_subclass = train_test_data['MSSubClass']

# Location and Zoning of Lot
zoning = pd.get_dummies(train_test_data['MSZoning'],prefix='MSZoning')
neighborhood = pd.get_dummies(train_test_data['Neighborhood'],prefix='Neighborhood')
sale_location = pd.concat([zoning,neighborhood],axis=1)

# Road Access of Lot
lotfrontage = train_test_data['LotFrontage'].replace('NA',1e-6)
lotfrontage = lotfrontage.replace(1e-6,lotfrontage.mean())
street = pd.get_dummies(train_test_data['Street'],prefix='Street')
alley = pd.get_dummies(train_test_data['Alley'],prefix='Alley')
sale_roadaccess = pd.concat([lotfrontage,street,alley],axis=1)

# Shape and Terrain of Lot
lotshape = pd.get_dummies(train_test_data['LotShape'],prefix='LotShape')
landcontour = pd.get_dummies(train_test_data['LandContour'],prefix='LandContour')
landslope = pd.get_dummies(train_test_data['LandSlope'],prefix='LandSlope')
lotconfig = pd.get_dummies(train_test_data['LotConfig'],prefix='LotConfig')
sale_property = pd.concat([train_test_data['LotArea'],lotshape,landcontour,landslope,lotconfig],axis=1)

# Proximity to Conditions
#train_test_data['Condition1'] = train_test_data['Condition1'].map(lambda t: t.replace('Brk Cmn','BrkComm').replace('CmentBd','CemntBd').replace('Wd Shng','Wd Sdng').replace('Other','VinylSd').replace('WdShing','NA'))
#train_test_data['Condition2'] = train_test_data['Condition2'].map(lambda t: t.replace('Brk Cmn','BrkComm').replace('CmentBd','CemntBd').replace('Wd Shng','Wd Sdng').replace('Other','VinylSd').replace('WdShing','NA'))
surroundings1 = pd.get_dummies(train_test_data['Condition1'])
surroundings2 = pd.get_dummies(train_test_data['Condition2'],columns=surroundings1.columns)
sale_surroundings = surroundings1+surroundings2
sale_surroundings['RRNe'] = train_test_data['Condition1'].map(lambda s: 1 if s=='RRNe' else 0)

# House Type, style and Condition
bldgtype = pd.get_dummies(train_test_data['BldgType'],prefix='BldgType')
housestyle = pd.get_dummies(train_test_data['HouseStyle'],prefix='HouseStyle')
roofstyle = pd.get_dummies(train_test_data['RoofStyle'],prefix='RoofStyle')
sale_housestyle = pd.concat([bldgtype,housestyle,roofstyle,train_test_data['OverallQual'],train_test_data['OverallCond']],axis=1)

# Year of Construction and Remodeling
houseage = train_test_data['YearBuilt'].map(lambda y: y-train_test_data['YearBuilt'].min())
remodage = train_test_data['YearRemodAdd'].map(lambda y: y-train_test_data['YearRemodAdd'].min())
sale_houseage = pd.concat([houseage,remodage],axis=1)

# External Materials
roofmatl = pd.get_dummies(train_test_data['RoofMatl'],prefix='RoofMatl')
train_test_data['Exterior1st'] = train_test_data['Exterior1st'].map(lambda t: t.replace('Brk Cmn','BrkComm').replace('CmentBd','CemntBd').replace('Wd Shng','Wd Sdng').replace('Other','VinylSd'))
train_test_data['Exterior2nd'] = train_test_data['Exterior2nd'].map(lambda t: t.replace('Brk Cmn','BrkComm').replace('CmentBd','CemntBd').replace('Wd Shng','Wd Sdng').replace('Other','VinylSd'))
exteriorcover1 = pd.get_dummies(train_test_data['Exterior1st'])
exteriorcover2 = pd.get_dummies(train_test_data['Exterior2nd'],columns=exteriorcover1.columns)
total_exteriorcover = exteriorcover1+exteriorcover2
total_exteriorcover['WdShing'] = train_test_data['Exterior1st'].map(lambda t: 1 if t=='WdShing' else 0)
exterqual = train_test_data['ExterQual'].map(QuScore_Dict)
extercond = train_test_data['ExterCond'].map(QuScore_Dict)
sale_extmatl = pd.concat([roofmatl,total_exteriorcover,exterqual,extercond],axis=1)

# Masonry Veneer
masvnrtype = pd.get_dummies(train_test_data['MasVnrType'],prefix='MasVnrType')
masvnrarea = train_test_data['MasVnrArea'].map(lambda m: 0 if m=='NA' else m)
sale_masvnr = pd.concat([masvnrtype,masvnrarea],axis=1)

# Foundation Material
sale_foundation = pd.get_dummies(train_test_data['Foundation'],prefix='Foundation')

# Basement
bsmtqual = train_test_data['BsmtQual'].map(QuScore_Dict)
bsmtcond = train_test_data['BsmtCond'].map(QuScore_Dict)
bsmtexposure = pd.get_dummies(train_test_data['BsmtExposure'],prefix='BsmtExposure')
train_test_data['BsmtFinSF1'] = train_test_data['BsmtFinSF1'].map(lambda b: 0 if b=='NA' else b)
train_test_data['BsmtFinSF2'] = train_test_data['BsmtFinSF2'].map(lambda b: 0 if b=='NA' else b)
train_test_data['BsmtUnfSF'] = train_test_data['BsmtUnfSF'].map(lambda b: 0 if b=='NA' else b)
bsmtfintype = train_test_data.apply(lambda df: BsmtFinType_Dict[df['BsmtFinType1']]*df['BsmtFinSF1']+BsmtFinType_Dict[df['BsmtFinType2']]*df['BsmtFinSF2']+df['BsmtUnfSF'],axis=1)
bsmtfintype = bsmtfintype.map(lambda b: 0 if b=='NA' else b)
sale_bsmt = pd.concat([bsmtqual,bsmtcond,bsmtexposure,bsmtfintype],axis=1)

# Utilities and Heating
utilities = pd.get_dummies(train_test_data['Utilities'],prefix='Utilities')
heating = pd.get_dummies(train_test_data['Heating'],prefix='Heating')
heatingqc = train_test_data['HeatingQC'].map(QuScore_Dict)
centralair = pd.get_dummies(train_test_data['CentralAir'],prefix='CentralAir')
electrical = pd.get_dummies(train_test_data['Electrical'],prefix='Electrical')
fireplace = train_test_data.apply(lambda df: df['Fireplaces']*QuScore_Dict[df['FireplaceQu']],axis=1)
sale_utilities = pd.concat([utilities,heating,heatingqc,centralair,electrical,fireplace],axis=1)

# Living Rooms
train_test_data['BsmtFullBath'] = train_test_data['BsmtFullBath'].map(lambda t: 0 if t=='NA' else t)
train_test_data['BsmtHalfBath'] = train_test_data['BsmtHalfBath'].map(lambda t: 0 if t=='NA' else t)
bathnum = train_test_data.apply(lambda df: df['FullBath']+df['HalfBath']*0.5+df['BsmtFullBath']*0.75+df['BsmtHalfBath']*0.25,axis=1)
livarea = train_test_data.apply(lambda df: df['1stFlrSF']+df['2ndFlrSF']+df['LowQualFinSF']*0.5,axis=1)
kitchen = train_test_data.apply(lambda df: df['KitchenAbvGr']*QuScore_Dict[df['KitchenQual']],axis=1)
functional = train_test_data['Functional'].map(Functional_Dict).fillna(0,inplace=True)
sale_livrooms = pd.concat([bathnum,livarea,train_test_data['BedroomAbvGr'],kitchen,functional],axis=1)

# Garage
garagetype = pd.get_dummies(train_test_data['GarageType'],prefix='GarageType')
train_test_data['GarageYrBlt'] = train_test_data['GarageYrBlt'].map(lambda t: 2011 if t=='NA' else t)
garageage = train_test_data['GarageYrBlt'].map(lambda t: train_test_data['GarageYrBlt'].min() if t==2011 else t).map(lambda y: y-train_test_data['GarageYrBlt'].min())
garagefinish = train_test_data['GarageFinish'].map(FinishScore_Dict)
train_test_data['GarageCars'] = train_test_data['GarageCars'].map(lambda t: 0 if t=='NA' else t)
train_test_data['GarageArea'] = train_test_data['GarageArea'].map(lambda t: 0 if t=='NA' else t)
avg_cars_garage = train_test_data['GarageCars'].loc[train_test_data['GarageCars'] != 0].mean()
avg_area_garage = train_test_data['GarageArea'].loc[train_test_data['GarageArea'] != 0].mean()
avg_area_per_car = avg_area_garage/avg_cars_garage
garagecars = train_test_data['GarageCars']
garageareaspacious = train_test_data.apply((lambda x: x['GarageArea']/avg_area_per_car-x['GarageCars']),axis=1)
garagequal = train_test_data['GarageQual'].map(QuScore_Dict)
garagecond = train_test_data['GarageCond'].map(QuScore_Dict)
paveddrive = train_test_data['PavedDrive'].map(PaveScore_Dict)
sale_garage = pd.concat([garagetype,garageage,garagefinish,garagecars,garageareaspacious,garagequal,garagecond,paveddrive],axis=1)

# Deck and Porch
sale_porch = pd.concat([train_test_data['WoodDeckSF'],train_test_data['OpenPorchSF'],train_test_data['EnclosedPorch'],train_test_data['3SsnPorch'],train_test_data['ScreenPorch']],axis=1)

# Pool
poolcond = train_test_data['PoolQC'].map(QuScore_Dict).fillna(0,inplace=True)
sale_pool = pd.concat([train_test_data['PoolArea'],poolcond],axis=1)

# Fence
sale_fence = train_test_data['Fence'].map(FenceScore_Dict).fillna(0,inplace=True)

# Miscellaneous Features
miscfeature = pd.get_dummies(train_test_data['MiscFeature'],prefix='MiscFeature')
sale_misc = pd.concat([miscfeature,train_test_data['MiscVal']],axis=1)

# Month and Year of Sale
sale_datesold = pd.concat([train_test_data['MoSold'],train_test_data['YrSold']],axis=1)

# Sale Condition and Type
saletype = pd.get_dummies(train_test_data['SaleType'],prefix='SaleType')
salecondition = pd.get_dummies(train_test_data['SaleCondition'],prefix='SaleCondition')
sale_typecond = pd.concat([saletype,salecondition],axis=1)

# ==========================
# Train/Test Data Definition
# ==========================
transformed_data = pd.DataFrame()
feature_set = [sale_subclass,sale_location,sale_roadaccess,sale_property,sale_surroundings,sale_housestyle,sale_houseage,sale_extmatl,sale_masvnr,\
sale_foundation,sale_bsmt,sale_utilities,sale_livrooms,sale_garage,sale_porch,sale_pool,sale_fence,sale_misc,sale_datesold,sale_typecond]
for i in range(len(feature_set)):
    transformed_data = pd.concat([transformed_data,feature_set[i]],axis=1)
#transformed_data.to_csv('transformed_data.csv', index=False)
#sys.exit(0)
train_X = transformed_data[:1460]
train_y = train_data['SalePrice']
test_X = transformed_data[1460:]
k_fold = KFold(n_splits=3)

# ========================================
# Linear Regression Modelling and Solution
# ========================================
alphas = np.logspace(-1,1,20)
l1_ratios = np.linspace(0,1,21)
prediction_model_lr = GridSearchCV(estimator=ElasticNet(fit_intercept=True,random_state=0,precompute=True),param_grid=dict(alpha=alphas,l1_ratio=l1_ratios),cv=k_fold,iid=False,n_jobs=-1)
lc.learning_curve_plot(prediction_model_lr,train_X,train_y)
sys.exit(0)
prediction_model_lr.fit(train_X[:-100], train_y[:-100])
print(cross_val_score(prediction_model_lr, train_X[-100:], train_y[-100:], cv=k_fold, n_jobs=-1))
prediction_y_lr_cv = prediction_model_lr.predict(train_X[-100:])
prediction_result_lr_cv = pd.DataFrame({'PredictionCV': prediction_y_lr_cv, 'GroundTruthCV': train_y[-100:]})
prediction_result_lr_cv.to_csv('prediction_submission_lr_cv.csv', index=False)
#print(prediction_model_lr.best_score_, prediction_model_lr.best_estimator_.alpha, prediction_model_lr.best_estimator_.l1_ratio)
#prediction_y_lr = prediction_model_lr.predict(test_X)
#prediction_result_lr = pd.DataFrame({'SalePrice': prediction_y_lr})
#prediction_result_lr.to_csv('prediction_submission_lr.csv', index=True)

# ===============================================
# Random Forest Regression Modelling and Solution
# ===============================================
n_values = np.arange(5,105,5)
prediction_model_rfr = GridSearchCV(estimator=RandomForestRegressor(random_state=0),param_grid=dict(n_estimators=n_values),cv=k_fold,iid=False,n_jobs=-1)
lc.learning_curve_plot(prediction_model_rfr,train_X,train_y)
sys.exit(0)
prediction_model_rfr.fit(train_X[:-100], train_y[:-100])
print(cross_val_score(prediction_model_rfr, train_X[-100:], train_y[-100:], cv=k_fold, n_jobs=-1))
prediction_y_rfr_cv = prediction_model_rfr.predict(train_X[-100:])
prediction_result_rfr_cv = pd.DataFrame({'PredictionCV': prediction_y_rfr_cv, 'GroundTruthCV': train_y[-100:]})
prediction_result_rfr_cv.to_csv('prediction_submission_rfr_cv.csv', index=False)
#print(prediction_model_rfr.best_score_, prediction_model_rfr.best_estimator_.n_estimators)
#prediction_y_rfr = prediction_model_rfr.predict(test_X)
#prediction_result_rfr = pd.DataFrame({'SalePrice': prediction_y_rfr})
#prediction_result_rfr.to_csv('prediction_submission_rfr.csv', index=True)

# ================================================
# Support Vector Regression Modelling and Solution
# ================================================
Cs = np.logspace(2,5,30)
#kernels = np.array(['linear','poly','rbf','sigmoid','precomputed'])
prediction_model_svr = GridSearchCV(estimator=SVR(gamma='auto'),param_grid=dict(C=Cs),cv=k_fold,iid=False,n_jobs=-1)
lc.learning_curve_plot(prediction_model_svr,train_X,train_y)
sys.exit(0)
prediction_model_svr.fit(train_X[:-100], train_y[:-100])
print(cross_val_score(prediction_model_svr, train_X[-100:], train_y[-100:], cv=k_fold, n_jobs=-1))
prediction_y_svr_cv = prediction_model_svr.predict(train_X[-100:])
prediction_result_svr_cv = pd.DataFrame({'PredictionCV': prediction_y_svr_cv, 'GroundTruthCV': train_y[-100:]})
prediction_result_svr_cv.to_csv('prediction_submission_svr_cv.csv', index=False)
#print(prediction_model_svr.best_score_, prediction_model_svr.best_estimator_.C, prediction_model_svr.best_estimator_.kernel)
#prediction_y_svr = prediction_model_svr.predict(test_X)
#prediction_result_svr = pd.DataFrame({'SalePrice': prediction_y_svr})
#prediction_result_svr.to_csv('prediction_submission_svr.csv', index=True)

# ================================================
# Neural Network Regression Modelling and Solution
# ================================================
layer_sizes = np.arange(10,200,10)
activations = np.array(['identity','logistic','tanh','relu'])
prediction_model_nnr = GridSearchCV(estimator=MLPRegressor(solver='lbfgs'),param_grid=dict(hidden_layer_sizes=layer_sizes,activation=activations),cv=k_fold,iid=False,n_jobs=-1)
lc.learning_curve_plot(prediction_model_nnr,train_X,train_y)
sys.exit(0)
prediction_model_nnr.fit(train_X[:-100], train_y[:-100])
print(cross_val_score(prediction_model_nnr, train_X[-100:], train_y[-100:], cv=k_fold, n_jobs=-1))
prediction_y_nnr_cv = prediction_model_nnr.predict(train_X[-100:])
prediction_result_nnr_cv = pd.DataFrame({'PredictionCV': prediction_y_nnr_cv, 'GroundTruthCV': train_y[-100:]})
prediction_result_nnr_cv.to_csv('prediction_submission_nnr_cv.csv', index=False)
#print(prediction_model_nnr.best_score_, prediction_model_nnr.best_estimator_.hidden_layer_sizes, prediction_model_nnr.best_estimator_.activation)
#prediction_y_nnr = prediction_model_nnr.predict(test_X)
#prediction_result_nnr = pd.DataFrame({'SalePrice': prediction_y_nnr})
#prediction_result_nnr.to_csv('prediction_submission_nnr.csv', index=True)

# ==================================================
# Gaussian Process Regression Modelling and Solution
# ==================================================
#kernels = np.array([RBF(), Matern(), WhiteKernel(), PairwiseKernel(), DotProduct(), CompoundKernel([RBF(),DotProduct()])])
#prediction_model_gpr = GridSearchCV(estimator=GaussianProcessRegressor(random_state=0),param_grid=dict(kernel=kernels),cv=k_fold,iid=False,n_jobs=-1)
#lc.learning_curve_plot(prediction_model_gpr,train_X,train_y)
#sys.exit(0)
#prediction_model_gpr.fit(train_X[:-100], train_y[:-100])
#print(cross_val_score(prediction_model_gpr, train_X[-100:], train_y[-100:], cv=k_fold, n_jobs=-1))
#prediction_y_gpr_cv = prediction_model_gpr.predict(train_X[-100:])
#prediction_result_gpr_cv = pd.DataFrame({'PredictionCV': prediction_y_gpr_cv, 'GroundTruthCV': train_y[-100:]})
#prediction_result_gpr_cv.to_csv('prediction_submission_gpr_cv.csv', index=False)
#print(prediction_model_gpr.best_score_, prediction_model_gpr.best_estimator_.kernel)
#prediction_y_gpr = prediction_model_gpr.predict(test_X)
#prediction_result_gpr = pd.DataFrame({'SalePrice': prediction_y_gpr})
#prediction_result_gpr.to_csv('prediction_submission_gpr.csv', index=True)

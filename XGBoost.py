from sklearn.metrics import mean_squared_error, r2_score, make_scorer
from sklearn.model_selection import RandomizedSearchCV, KFold 
from xgboost import XGBRegressor 
from sklearn.feature_selection import RFE 
from sklearn.pipeline import Pipeline 

building_IDs=list(range(1, 368))
upload_data=pd.DataFrame()
for i in building_IDs:
    input_data=training_data[training_data["ID"]==i]
    input_data.dropna ( axis=0 , inplace =True )
    X_df=input_data.drop(['eload','Month', 'Hour', 'Day of Week','ID'], axis=1)
    X=X_df.values # Independent variables (Features) 
    y=input_data['eload'].values # Dependent variable

    np.random.seed(123)
    kf=KFold(n_splits=5, shuffle=True )
    scoring_param=make_scorer (mean_squared_error, greater_is_better=False)
    num_features_to_select=np.arange(1,20,1).tolist()
    num_features_to_select.sort(reverse=True)
    p_grid_boost= dict(FS__n_features_to_select = num_features_to_select, 
                       XGB__max_depth = [int(i) for i in np.linspace( 1, 20, num=10)], 
                       XGB__learning_rate=np.linspace( 0.001,0.1, num=25), 
                       XGB__n_estimators=[int(i) for i in np.linspace(100,1000, num=25)] )

    #XGBoost model training
    estimators=[]
    estimators.append (('FS',RFE( estimator=XGBRegressor(n_jobs=4))))
    estimators.append (('XGB', XGBRegressor(n_jobs =4)))
    model=Pipeline(estimators)
    XBoost_grid_model=RandomizedSearchCV(estimator=model, param_distributions=p_grid_boost, n_iter=25,scoring=scoring_param, cv=kf)
    XBoost_grid_model.fit(X, y)

    baseline_train = XBoost_grid_model.predict(X)
    #Input model information 
    p=input_data.shape[1]
    n=input_data.shape[0]
    print('Building ID {}, has {} training samples.'.format(i, n))

    rmse_train = math.sqrt(mean_squared_error(y,baseline_train))
    ndb_train= ((baseline_train.sum()/y.sum())-1)*100

    r2_train=r2_score(y,baseline_train)
    adj_r2_train = 1-(1-r2_train)*(n-1)/(n-p-1)

    print('Linear model results for building ID: {}\nCalibration set - R2 = {:.3f},  RMSE = {:.3f} C, CV(RMSE) = {:.2f} %, Net Determination Bias = {:.14f} %'.format(i, adj_r2_train,rmse_train, 100*math.sqrt(mean_squared_error(y,baseline_train))/y.mean(),ndb_train))

    prediction_data=pd.DataFrame()
    prediction_data =test_data[test_data["ID"]==i]
    
    X_test_df=prediction_data.drop(['Month', 'Hour', 'Day of Week','ID'], axis=1)
    X_test=X_test_df.values # Independent variables (Features) 

    print('Building ID {}, has {} testing samples, predictng.'.format(i, prediction_data.shape[0]))
    prediction_data['KWH'] = XBoost_grid_model.predict(X=X_test)

    upload_data_temp=prediction_data[['KWH', 'ID']]
    upload_data=upload_data.append(upload_data_temp)

#Save results as CSV
data_path = os.path.join(os.getcwd(), 'EVO_Download', 'XGBoost_model_test_.csv')
upload_data.to_csv(data_path)

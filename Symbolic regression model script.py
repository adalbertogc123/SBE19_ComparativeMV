import math
from sklearn.metrics import mean_squared_error, r2_score
from gplearn.genetic import SymbolicRegressor

function_set = ['add', 'sub', 'mul','div', 'sqrt','log','abs','inv','neg','cos','sin','tan']
building_IDs=list(range(1, 368))
upload_data=pd.DataFrame()

for i in building_IDs:
    input_data=training_data[training_data["ID"]==i]
    X_df=input_data.drop(['eload','Month', 'Hour', 'Day of Week','ID'], axis=1)
    X=X_df.values # Independent variables (Features) 
    y=input_data['eload'].values # Dependent variable
    max_value=float(input_data['eload'].quantile(q=0.95))
    min_value=-1*max_value
    est_gp= SymbolicRegressor(const_range=(min_value, max_value),population_size=12000,
                              generations=20, stopping_criteria=0.1,
                              p_crossover=0.7, p_subtree_mutation=0.1,
                              p_hoist_mutation=0.05, p_point_mutation=0.1,
                              max_samples=0.9, verbose=0,
                              parsimony_coefficient=0.001,
                              random_state=0,metric='mean absolute error',
                              function_set=function_set,n_jobs=4,warm_start=False)
    est_gp.fit(X, y)
    baseline_train = est_gp.predict(X=X)

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

    print('Building ID {}, has {} testing samples. Predicting...'.format(i, prediction_data.shape[0]))
    prediction_data['KWH'] = est_gp.predict(X=X_test)
    upload_data_temp=prediction_data[['KWH', 'ID']]
    upload_data=upload_data.append(upload_data_temp)

#Save results as CSV
data_path = os.path.join(os.getcwd(), 'EVO_Download', 'Symbolic_regression_model_test.csv')
upload_data.to_csv(data_path)

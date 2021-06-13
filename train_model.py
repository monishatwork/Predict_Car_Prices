import pandas as pd 
from sklearn.preprocessing import LabelEncoder
from functools import partial
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import statsmodels.api as sm
import utility 
from functools import partial
import clean_data



numeric_transform_vars = ['price','horsepower','carvolume','mpg_enginsize_ratio','compressionratio',\
                          'wheelbase','peakrpm','enginesize']

def create_reg_model(df=None,target_var=None,model_params=None):
#     print(model_params)
    X_train = df.drop(target_var,axis='columns')
    y_train = df[target_var]
    X_train_ = sm.add_constant(X_train[model_params].to_numpy())
    lr_model = sm.OLS(y_train, X_train_).fit()
    print(lr_model.summary())
    return {'model_features':X_train[model_params],'model':lr_model}

def make_prediction(df=None, features=None, tar_var=None, model=None):
    y_actual = df[tar_var]
    df_ = sm.add_constant(df[features])
    y_pred = model.predict(df_)
    return y_actual, y_pred


def create_model_dataset(model_data):
    train, test = train_test_split(model_data, train_size = 0.7, test_size = 0.3, random_state = 100)    
    scaler = MinMaxScaler()
    train.loc[:, numeric_transform_vars] = scaler.fit_transform(train[numeric_transform_vars])
    return train, test, scaler


def export_model(model):
    # Save the model
    pkl_path = 'model.pkl'
    with open(pkl_path, 'wb') as file:
        pickle.dump(model, file)
        print(f"Model saved at {pkl_path}")



train, test, scaler = create_model_dataset(clean_data.preprocess_data())
model_1_params = ['horsepower']
reg_model = partial(utility.create_reg_model, train, 'price')
model_1 = reg_model(model_1_params)

y_actual, y_pred = make_prediction(df=train, features=model_1_params, \
                                  tar_var='price', model=model_1['model'])


utility.test_model(test, 'price', numeric_transform_vars, model_1, scaler)


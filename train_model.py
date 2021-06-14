import pandas as pd 
from sklearn.linear_model import LinearRegression
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from functools import partial
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, StandardScaler
import statsmodels.api as sm
import utility 
from functools import partial
import clean_data
import pickle


def create_reg_model(df=None, target_var=None, model_params=None):
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


def create_train_model_pipeline(model_data):
    '''
    input: cleaned data
    returns: model    
    
    creates the pipeline out of the model_data input, splits the model_data
    into train and test, perfoms transformation on the data, supplies the 
    data to the model, creates the model.
    '''

    X = model_data.drop('price', axis='columns')
    y = model_data['price']
    
    numeric_features = ['horsepower','carvolume','mpg_enginsize_ratio','compressionratio',\
                          'wheelbase','peakrpm','enginesize']
    
    numeric_transformer = Pipeline(steps=[('min_max', MinMaxScaler())])

    categorical_features = [cat_cols for cat_cols in model_data.select_dtypes(include='object').columns]
    categorical_transformer = OneHotEncoder(handle_unknown='ignore')

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)])
    
    regresor = Pipeline(steps=[('preprocessor', preprocessor),
                      ('lr', LinearRegression())])
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
    
    regresor.fit(X_train, y_train)
    
    return regresor, X, y


def export_model(model):
    '''
    input: trained model
    returns: pickel file of the model
    exports trained model and its weigths.
    '''

    pkl_path = 'model.pkl'
    with open(pkl_path, 'wb') as file:
        pickle.dump(model, file)
        print(f"Model saved at {pkl_path}")


cleaned_data = utility.read_data("cleaned_data.csv")
model, X, y = create_train_model_pipeline(cleaned_data)

utility.test_model(model, X, y)


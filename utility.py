import pandas as pd 
from sklearn.preprocessing import LabelEncoder
import statsmodels.api as sm

def read_data(filename):
    df = pd.read_csv(filename)
    return df

# def get_meta_data(df,filename):
#     with open("car_pricing_metadata.txt", 'w') as f:
#         f.write(cap.stdout)
#     print(create_meta_data_file(df, filename))


def transform_data(df):
    le = LabelEncoder()
    col_trns=[cat_cols for cat_cols in df.select_dtypes(include='object').columns]
        
    for col in col_trns:
        col_name = 'enc_'+col
        df.loc[:,col_name] = le.fit_transform(df[col])

def create_dummy_vars(df, col_name, prefix_name, prefix_seprtr):
    df_dummy=pd.get_dummies(df[col_name], prefix=prefix_name, prefix_sep=prefix_seprtr, drop_first=True)
#     print(tabulate(df_dummy, headers='keys', tablefmt="grid"))
    df = pd.concat([df,df_dummy], axis='columns')
    return df
    
    

def create_plots():
    pass

def perform_univariate_analysis():
    pass

def calculate_prcnt(group):
    return str(round(((group).sum()/(car_pricing['price'].sum()))*100,2))+'%'

def calculate_crosstab_percnt(cols=None, rows=None, df=None):
    pvt_tbl = pd.pivot_table(data=df,index=rows, columns=cols, aggfunc=calculate_prcnt, \
                             fill_value='NA')
    return pvt_tbl

def create_reg_model(df=None,target_var=None,model_params=None):
#     print(model_params)
    X_train = df.drop(target_var,axis='columns')
    y_train = df[target_var]
    X_train_ = sm.add_constant(X_train[model_params].to_numpy())
    lr_model = sm.OLS(y_train, X_train_).fit()
    print(lr_model.summary())
    return {'model_features':X_train[model_params],'model':lr_model}

def calculate_vif(X):
    vif = pd.DataFrame()
    vif["VIF Factor"] = [variance_inflation_factor(X.values, X_val) for X_val in range(X.shape[1])]
    vif["features"] = X.columns
    print(tabulate(vif, headers='keys', tablefmt="grid"))

def evaluate_model():
    pass

def get_model_diagnostic_plots(y_actual, y_predicted):
    residual = y_actual - y_predicted
    fig, ax = plt.subplots(2,2,figsize=(16,10), constrained_layout=True)
    sns.residplot(y_actual, y_predicted, lowess=True, color="blue", ax=ax[0][0]);
    sns.scatterplot(y_actual, y_predicted, color="blue", ax=ax[0][1]);
    sp.stats.probplot(residual, plot=ax[1][0], fit=True);
    sns.distplot(residual, ax=ax[1][1])
#     fig.delaxes(ax[1][1])

def test_model(df, target_var, numeric_transform_vars, model, transfomer):
    df.loc[:, numeric_transform_vars] = transfomer.transform(df.loc[:,numeric_transform_vars])
    X_test = df.drop(target_var, axis='columns')
    y_actual = df[target_var]
    
    X_test_ = sm.add_constant(X_test)
    y_predicted  = model.predict(X_test_)
    return y_actual, y_predicted

def make_prediction(df=None, features=None, tar_var=None, model=None):
    y_actual = df[tar_var]
    df_ = sm.add_constant(df[features])
    y_pred = model.predict(df_)
    return y_actual, y_pred


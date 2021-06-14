import utility
import os
from pandas import read_csv

## Read Data
def read_data(filename):
    df = read_csv(filename)
    return df


## Preprocess Data
def preprocess_data(filename):
    df = read_data(filename)
    print(f'Before Cleaning: {df.shape}')
    df.loc[:,'Car_Compny'] = df.CarName.apply(lambda car_cmpny: car_cmpny.split(' ')[0])
    df['carvolume'] = df['carheight']*df['carwidth']*df['carlength']
    df['mpg_enginsize_ratio'] = df['citympg']/df['enginesize']
    df.loc[df.carbody.isin(['convertible','hardtop']),'car_body_new_cats'] = 'carbody_a'
    df.loc[~(df.carbody.isin(['convertible','hardtop'])),'car_body_new_cats']= 'carbody_b'

    df.loc[df.enginetype=='dohc','enginetype_new_cats']='eng_a'
    df.loc[df.enginetype=='ohcv','enginetype_new_cats']='eng_b'
    df.loc[df.enginetype=='dohcv','enginetype_new_cats']='eng_c'
    df.loc[~df.enginetype.isin(['dohcv','ohcv','dohc']),'enginetype_new_cats']='eng_d'
    df.loc[df.cylindernumber.isin(['two','three','four']),'cylindernumber_new_cats']='cno_a'
    df.loc[df.cylindernumber.isin(['five','six']),'cylindernumber_new_cats']='cno_b'
    df.loc[df.cylindernumber.isin(['twelve','eight']),'cylindernumber_new_cats']='cno_c'
    df.drop(['enginetype', 'cylindernumber', 'carbody'], axis='columns')
    df.to_csv('cleaned_data.csv', index=None)
    return df


if __name__ == "__main__":
    # Reads the file train.csv
    train_file = os.path.join('car_price_data.csv')

    if os.path.exists(train_file):
        cleaned_df = preprocess_data(train_file)
        print(f'After Cleaning: {cleaned_df.shape}')
    else:
        print(f'File not found {train_file}')
import pandas as pd
import numpy as np

"""
A set of helper functions for the Kaggle/Zillow case study for Data Science Dream Job
"""



def read_in_dataset(dset, verbose=False):
    
    """Read in one of the Zillow datasets (train or properties)

    Keyword arguments:
    dset -- a string in {properties_2016, properties_2017, train_2016, train_2017}
    verbose -- whether or not to print info about the dataset
    
    Returns:
    a pandas dataframe
    """
    
    df = pd.read_csv('unzipped_data/{0}.csv'.format(dset))
    
    if verbose:
        print('\n{0:*^80}'.format(' Reading in the {0} dataset '.format(dset)))
        print("\nit has {0} rows and {1} columns".format(*df.shape))
        print('\n{0:*^80}\n'.format(' It has the following columns '))
        print(df.columns)
        print('\n{0:*^80}\n'.format(' The first 5 rows look like this '))
        print(df.head())
        
    return df


def merge_dataset(train, properties):
    
    """Merge the train and properties datasets. Both need to have a common key `parcelid`

    Keyword arguments:
    train -- the dataframe of transactions
    properties -- the dataframe of properties
    
    Returns:
    a pandas dataframe
    """

    train_data_merged = train.merge(properties, how='left', on='parcelid')
    
    return train_data_merged



def filter_duplicate_parcels(df, random_state=0):
    """filter the merged train and properties datasets to only include one record per parcel.
    
    Intended only for use on the training data for building the model

    Keyword arguments:
    df -- the result of `merge_dataset`
    random_state -- the random seed to be passed to the `pandas.DataFrame.sample()` method
    
    Returns:
    a pandas dataframe
    """
 
    counts_per_parcel = df.groupby('parcelid').size()
    more_than_one_sale = df[df.parcelid.isin(counts_per_parcel[counts_per_parcel > 1].index)]
    only_one_sale = df[df.parcelid.isin(counts_per_parcel[counts_per_parcel == 1].index)]
    reduced_df = more_than_one_sale.sample(frac=1, random_state=random_state).groupby('parcelid').head(1)
    reduced_df = pd.concat([only_one_sale, reduced_df])
    
    return reduced_df


def get_data(dset):
    
    """Create the training dataset (2016) or the test dataset (2017)

    Keyword arguments:
    dset -- a string in {train, test}
    
    Returns:
    a tuple of pandas dataframe (X) and pandas series (y)
    """
    
    year = {'train':2016, 'test':2017}[dset]
    
    train = read_in_dataset('train_{0}'.format(year))
    properties = read_in_dataset('properties_{0}'.format(year))
    merged = merge_dataset(train, properties)
    
    if dset == 'train':
        merged = filter_duplicate_parcels(merged)
    
    y = merged.pop('logerror')
    return merged, y

def mean_abs_error(y_true, y_pred):
    return np.mean(np.abs(y_true - y_pred))
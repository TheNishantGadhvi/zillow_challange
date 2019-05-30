import pandas as pd

class preprocessor:
    
    def __init__(self, cols_to_filter=None, datecols=None):
        
        self.cols_to_filter = cols_to_filter
        self.datecols = datecols
        self.was_fit = False
    
    def fit(self, X, y=None):
        """learn any information from the training data we may need to transform the test data"""
        
        # learn from the training data and return the class itself. 
        # allows you to chain fit and predict methods like 
        
        # > p = preprocessor()
        # > p.fit(X).transform(X)
        
        self.was_fit = True
        
        # filter
        X_new = X.drop(self.cols_to_filter, axis=1)
        
        categorical_features = X_new.dtypes[X_new.dtypes == 'object'].index
        self.categorical_features = [x for x in categorical_features if 'date' not in x]
        
        dummied = pd.get_dummies(X_new, columns=self.categorical_features, dummy_na=True)
        self.colnames = dummied.columns
        del dummied
        
        return self
    
    def transform(self, X, y=None):
        """transform the training or test data"""
        # transform the training or test data based on class attributes learned in the `fit` step
        
        if not self.was_fit:
            raise Error("need to fit preprocessor first")
        
        # filter
        X_new = X.drop(self.cols_to_filter, axis=1)

        # dummy code
        X_new = pd.get_dummies(X_new, columns=self.categorical_features, dummy_na=True)
        newcols = set(self.colnames) - set(X_new.columns)
        for x in newcols:
            X_new[x] = 0
            
        X_new = X_new[self.colnames]
                
        # fill NA after we dummy code
        X_new = X_new.fillna(-1)
        
        if self.datecols:
            for x in self.datecols:
                X_new[x + '_month'] = pd.to_datetime(X_new[x]).apply(lambda x: x.month)
                X_new[x + '_year'] = pd.to_datetime(X_new[x]).apply(lambda x: x.year)
                X_new = X_new.drop(x, axis=1)
        
        return X_new
    
    def fit_transform(self, X, y=None):
        """fit and transform wrapper method, used for sklearn pipeline"""

        return self.fit(X).transform(X)
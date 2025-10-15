import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder

class DataProcessor:
    """
    A class to handle all data preprocessing steps including cleaning,
    imputation, outlier handling, and encoding.
    """
    def __init__(self):
        self.imputers = {}
        self.label_encoders = {}
        self.outlier_bounds = {}
        self.numeric_cols = []
        self.categorical_cols = []
        self.target_encoder = None

    def fit(self, df, target_column):
        """
        Learn the preprocessing parameters from the training data.
        """
        # Drop the target column for fitting processors on features
        X = df.drop(columns=[target_column])
        y = df[target_column]

        # Identify column types
        self.numeric_cols = X.select_dtypes(include=np.number).columns.tolist()
        self.categorical_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()

        # 1. Handle Missing Values
        for col in self.numeric_cols:
            imputer = SimpleImputer(strategy='mean')
            self.imputers[col] = imputer.fit(X[[col]])
        for col in self.categorical_cols:
            imputer = SimpleImputer(strategy='most_frequent')
            self.imputers[col] = imputer.fit(X[[col]])

        # 2. Handle Outliers (store bounds)
        temp_df_numeric = X[self.numeric_cols].copy()
        for col in self.numeric_cols: # Impute before calculating quantiles
            temp_df_numeric[col] = self.imputers[col].transform(temp_df_numeric[[col]])

        for col in self.numeric_cols:
            Q1 = temp_df_numeric[col].quantile(0.25)
            Q3 = temp_df_numeric[col].quantile(0.75)
            IQR = Q3 - Q1
            lower = Q1 - 1.5 * IQR
            upper = Q3 + 1.5 * IQR
            self.outlier_bounds[col] = (lower, upper)

        # 3. Encode Categorical Features
        for col in self.categorical_cols:
            le = LabelEncoder()
            # Impute before fitting encoder
            col_imputed = self.imputers[col].transform(X[[col]])
            self.label_encoders[col] = le.fit(col_imputed.ravel())

        # 4. Encode Target Column
        self.target_encoder = LabelEncoder()
        self.target_encoder.fit(y)

        return self

    def transform(self, df, is_target=False):
        """
        Apply the learned transformations to new data.
        """
        df_copy = df.copy()

        if is_target:
            return pd.Series(self.target_encoder.transform(df_copy), name=df_copy.name)

        # Apply transformations
        for col in df_copy.columns:
            if col in self.imputers:
                df_copy[col] = self.imputers[col].transform(df_copy[[col]]).ravel()
            if col in self.outlier_bounds:
                lower, upper = self.outlier_bounds[col]
                df_copy[col] = np.where(df_copy[col] < lower, lower,
                                       np.where(df_copy[col] > upper, upper, df_copy[col]))
            if col in self.label_encoders:
                # Handle unseen labels in prediction data by assigning a default value (e.g., -1 or a new class)
                df_copy[col] = df_copy[col].astype(str).map(lambda s: self.label_encoders[col].transform([s])[0] if s in self.label_encoders[col].classes_ else -1)

        return df_copy

    def fit_transform(self, df, target_column):
        """
        Fit on data and then transform it.
        """
        self.fit(df, target_column)
        
        X = df.drop(columns=[target_column])
        y = df[target_column]

        X_transformed = self.transform(X)
        y_transformed = self.transform(y, is_target=True)

        return X_transformed, y_transformed
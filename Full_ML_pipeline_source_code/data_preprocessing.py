import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer

#======================================================================================================
class TelcoDataProcessor:
    def __init__(self, file_path):
        self.customer_details = pd.read_csv(file_path)
        self.df = self.customer_details.copy()
        self.feature_names = []

    # Call once per fold, and once more for the final full-train fit.
    @staticmethod
    def make_preprocessing_state():
        return {"imputer": SimpleImputer(strategy="median"),
                "encoder": OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                "feature_names": []
                }

    def clean_data(self):
        self.df.drop('customerID', axis=1, inplace=True)
        self.df['TotalCharges'] = pd.to_numeric(self.df['TotalCharges'], errors='coerce')
        self.df.replace(['No internet service', 'No phone service'], 'No', inplace=True)

        service_cols = ["PhoneService", "MultipleLines", "InternetService","OnlineSecurity", "OnlineBackup",
                        "DeviceProtection","TechSupport", "StreamingTV", "StreamingMovies"]
        existing_services = [c for c in service_cols if c in self.df.columns]
        # Feature Engineering
        self.df["HasMultipleServices"] = (self.df[existing_services] == "Yes").sum(axis=1)

        yes_no_cols = ['Partner', 'Dependents', 'PhoneService', 'MultipleLines','OnlineSecurity', 'OnlineBackup',
                       'DeviceProtection','TechSupport', 'StreamingTV','StreamingMovies','PaperlessBilling', 'Churn']
        for col in yes_no_cols:
            self.df[col] = self.df[col].map({'Yes': 1, 'No': 0})
        self.df['gender'] = self.df['gender'].map({'Female': 1, 'Male': 0})
        print("Shape after data cleaning: " + str(self.df.shape))

        return self.df

    def _apply_pipeline(self, df, state, is_fit=False):
        df = df.copy()
        # 1. Impute
        if is_fit: df['TotalCharges'] = state['imputer'].fit_transform(df[['TotalCharges']]).ravel() # for train data
        else: df['TotalCharges'] = state['imputer'].transform(df[['TotalCharges']]).ravel() # for test data

        # 2. Feature Engineering
        df["AverageMonthlyCharge"] = df["TotalCharges"] / (df["tenure"] + 1)

        # 3. One-Hot Encode
        categorical_cols = ['InternetService', 'Contract', 'PaymentMethod']
        if is_fit: encoded = state['encoder'].fit_transform(df[categorical_cols])
        else: encoded = state['encoder'].transform(df[categorical_cols])

        encoded_df = pd.DataFrame(encoded, columns=state['encoder'].get_feature_names_out(categorical_cols),
                                  index=df.index)

        df = df.drop(columns=categorical_cols)
        df = pd.concat([df, encoded_df], axis=1)

        if is_fit:
            state['feature_names'] = list(df.columns)
            self.feature_names = state['feature_names']
        else: df = df[state['feature_names']]  # Force column order

        return df, state

    def fit_transform_preprocessing(self, x_train_df, state=None):
        if state is None: state = self.make_preprocessing_state()
        return self._apply_pipeline(x_train_df, state, is_fit=True)

    def transform_preprocessing(self, x_test_df, state):
        df, _ = self._apply_pipeline(x_test_df, state, is_fit=False)
        return df

    @staticmethod
    def apply_smote(x_train, y_train, random_state=42):
        smote = SMOTE(random_state=random_state)
        x_train_res, y_train_res = smote.fit_resample(x_train, y_train)
        x_train_res = x_train_res.astype(np.float32)
        y_train_res = y_train_res.astype(np.int32)
        x_train_res = x_train_res.reshape(x_train_res.shape[0], x_train_res.shape[1], 1)
        after_sampling = {"Non-Churn": int((y_train_res == 0).sum()),"Churn": int((y_train_res == 1).sum())}
        return x_train_res, y_train_res, after_sampling

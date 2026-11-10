import pandas as pd
import warnings
warnings.filterwarnings("ignore")

# -------------------------------Data Analysis ------------------------------
eda_plot_df,num_cols,cat_cols= None,None,None
def data_analysis(file_path):
    # 1. Load the dataset
    df = pd.read_csv(file_path)

    print("Number of customers, Number of columns/features: " + str(df.shape))
    print("\nDataset features data types:")
    print(df.info())

    #print("Check missing values features:\n" + str(df.isnull().sum()))
    print("\nDataset first 5 rows:\n" + str(df.head()))
    print("\nDataset bottom 5 rows:\n" + str(df.tail()))

    df.drop(columns=['customerID'], inplace=True)
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    df['TotalCharges'] = df['TotalCharges'].fillna(df['TotalCharges'].median())

    numerical_cols = df.select_dtypes(include=["int64", "float64"]).columns.tolist()
    categorical_cols = df.select_dtypes(include=["object"]).columns.tolist()
    print("\nNumerical columns:\n" + str(numerical_cols))
    print("\nCategorical columns:\n" + str(categorical_cols))
    print("\nDataset features unique values:\n" + str(df.nunique()))
    print("\nDataset summary statistics of numerical columns:\n" + str(df.describe()))

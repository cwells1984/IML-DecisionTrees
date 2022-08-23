import pandas as pd
import numpy as np
import preprocessing


# Read in the values for abalone and output as a Pandas Dataframe
# Label: 'Rings' column
def prepare_abalone(csv_file, display_mode=False):

    # Declare the column names
    col_names = ["Sex",
                 "Length",
                 "Diameter",
                 "Height",
                 "Whole weight",
                 "Shucked weight",
                 "Viscera weight",
                 "Shell weight",
                 "Rings"]

    # Read the csv into the dataframe
    df = pd.read_csv(csv_file, names=col_names)

    if display_mode:
        print("Before One-Hot Encoding:")
        print(df.head())
        print("")

    if display_mode:
        print("After One-Hot Encoding:")
        print(df.head())
        print("")

    # Return the constructed Dataframe
    return df


# Read in the values for breast_cancer_wisconsin and output as a Pandas Dataframe
# Label: 'Class' column
# Feature Notes:
# 'Bare Nuclei' contains missing values denoted by '?', which this function replaces with NaN values
def prepare_breast_cancer_wisconsin(csv_file, display_mode=False):

    # Declare the column names
    col_names = ["Sample code number",
                 "Clump Thickness",
                 "Uniformity of Cell Size",
                 "Uniformity of Cell Shape",
                 "Marginal Adhesion",
                 "Single Epithelial Cell Size",
                 "Bare Nuclei",
                 "Bland Chromatin",
                 "Normal Nucleoli",
                 "Mitoses",
                 "Class"]

    # Read the csv into the dataframe
    df = pd.read_csv(csv_file, names=col_names)

    # Drop the Sample code number
    df.drop(columns=['Sample code number'], inplace=True)

    # Show missing data if in display mode
    if display_mode:
        print("Data types:")
        print(df.dtypes)
        n_missing = len(df.loc[df['Bare Nuclei'] == '?'])
        n_null = df['Bare Nuclei'].isnull().any()
        print(f'{n_missing} ? values in Bare Nuclei')
        print(f'Are null values in Bare Nuclei? {n_null}\n')

    # Replace the '?' missing values with NaN
    df.replace("?", np.nan, inplace=True)

    # Per the .names file, all the columns will have integer values
    # So, convert all the columns to the numpy Int64 data type to allow for NaN values in the column
    for col in df.columns:
        df[col] = df[col].astype('Int64')

    # Show missing data if in display mode - all columns should be ints and there should be null values
    if display_mode:
        print("Data types:")
        print(df.dtypes)
        n_missing = len(df.loc[df['Bare Nuclei'] == '?'])
        n_null = df['Bare Nuclei'].isnull().any()
        print(f'{n_missing} ? values in Bare Nuclei')
        print(f'Are null values in Bare Nuclei? {n_null}\n')

    # Pre-processing steps
    preprocessing.impute_missing_vals_in_column_int64(df, 'Bare Nuclei')

    # Show missing data if in display mode - there should be no more null values
    if display_mode:
        n_null = df['Bare Nuclei'].isnull().any()
        print(f'Are null values in Bare Nuclei? {n_null}\n')

    # Return the constructed Dataframe
    return df


# Read in the values for car and output as a Pandas Dataframe
# Label: 'CAR' column
def prepare_car(csv_file):

    # Declare the column names
    col_names = ["buying",
                 "maint",
                 "doors",
                 "persons",
                 "lug_boot",
                 "safety",
                 "CAR"]

    # Read the csv into the dataframe
    df = pd.read_csv(csv_file, names=col_names)

    # Return the constructed Dataframe
    return df


# Read in the values for forestfires and output as a Pandas Dataframe
# The column headers are included in the data file and do not need to be specified here
# Label: 'area' column
def prepare_forestfires(csv_file):

    # Read the csv into the dataframe
    df = pd.read_csv(csv_file)

    # Return the constructed Dataframe
    return df


# Read in the values for house-votes-84 and output as a Pandas Dataframe
# Label: 'party' column
def prepare_house_votes_84(csv_file):

    # Declare the column names
    col_names = ["party",
                 "handicapped-infants",
                 "water-project-cost-sharing",
                 "adoption-of-the-budget-resolution",
                 "physician-fee-freeze",
                 "el-salvador-aid",
                 "religious-groups-in-schools",
                 "anti-satellite-test-ban",
                 "aid-to-nicaraguan-contras",
                 "mx-missile",
                 "immigration",
                 "synfuels-corporation-cutback",
                 "education-spending",
                 "superfund-right-to-sue",
                 "crime",
                 "duty-free-exports",
                 "export-administration-act-south-africa"]

    # Read the csv into the dataframe
    df = pd.read_csv(csv_file, names=col_names)

    # Return the constructed Dataframe
    return df


# Read in the values for machine and output as a Pandas Dataframe
# Label: 'PRP' column
def prepare_machine(csv_file):

    # Declare the column names
    col_names = ["vendor name",
                 "Model Name",
                 "MYCT",
                 "MMIN",
                 "MMAX",
                 "CACH",
                 "CHMIN",
                 "CHMAX",
                 "PRP",
                 "ERP"]

    # Read the csv into the dataframe
    df = pd.read_csv(csv_file, names=col_names)

    # Pre-processing steps
    df.drop(columns=['Model Name'], inplace=True)

    # Return the constructed Dataframe
    return df

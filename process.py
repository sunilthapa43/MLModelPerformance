import pandas as pd
import numpy as np
from constants import keep_features, DoS_Types, Brute_Force_Types, Web_Attack_types, Others, usnw_encode_features, \
    iot_ton_encode_features, nsl_kdd_encode_features, column_names
from config import CICIDS_ROOT, TON_IOT_ROOT, USNW_ROOT, NSL_KDD_ROOT
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder


def get_final_df():
    # Load the dataset
    df1 = pd.read_csv(CICIDS_ROOT + 'Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv', encoding='latin1',
                      low_memory=False)
    df2 = pd.read_csv(CICIDS_ROOT + 'Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv',
                      encoding='latin1',
                      low_memory=False)
    df3 = pd.read_csv(CICIDS_ROOT + 'Friday-WorkingHours-Morning.pcap_ISCX.csv', encoding='latin1',
                      low_memory=False)
    df4 = pd.read_csv(CICIDS_ROOT + 'Monday-WorkingHours.pcap_ISCX.csv', encoding='latin1',
                      low_memory=False)
    df5 = pd.read_csv(CICIDS_ROOT + 'Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv',
                      encoding='latin1', low_memory=False)
    df6 = pd.read_csv(CICIDS_ROOT + 'Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv',
                      encoding='latin1', low_memory=False)
    df7 = pd.read_csv(CICIDS_ROOT + 'Tuesday-WorkingHours.pcap_ISCX.csv', encoding='latin1',
                      low_memory=False)
    df8 = pd.read_csv(CICIDS_ROOT + 'Wednesday-workingHours.pcap_ISCX.csv', encoding='latin1',
                      low_memory=False)

    ## Creating Our DataFrame
    # Merge the DataFrames horizontally based on common columns (all columns)
    combined_df = pd.concat([df1, df2, df3, df4, df5, df6, df7, df8], axis=0)
    combined_df.head()

    # Remove spaces before the first letter of column names
    combined_df = combined_df.rename(columns=lambda x: x.strip())

    # The features that are related to Electric Vehicle Charging Stations are kept and the other features are removed
    # Remove the features not in 'keep_features'
    Final_df = combined_df.drop(columns=[col for col in combined_df.columns if col not in keep_features])
    final_column_names = Final_df.columns.tolist()
    print('number of selected features: ', len(final_column_names) - 1)

    # Getting the categories of the Labels
    print('Labels are: ', Final_df['Label'].unique().tolist())
    print('The number of categories is: ', Final_df['Label'].nunique())

    Final_df['Label'] = Final_df['Label'].apply(lambda x: 'Dos' if x in DoS_Types else x)
    Final_df['Label'] = Final_df['Label'].apply(lambda x: 'Brute_Force' if x in Brute_Force_Types else x)
    Final_df['Label'] = Final_df['Label'].apply(lambda x: 'Web_Attack' if x in Web_Attack_types else x)
    Final_df['Label'] = Final_df['Label'].apply(lambda x: 'Bot/Infiltration/Heartbleed' if x in Others else x)
    return Final_df

def clean_df(df):
    # Create a stratified sample
    df = df.groupby('Label', group_keys=False).apply(lambda x: x.sample(frac=0.05))
    ## Preprocessings
    # Encoding the features
    features_to_encode = ['Flow ID', 'Source IP', 'Destination IP', 'Protocol', 'Label']
    encoder_dict = {}

    for feature in features_to_encode:
        le = LabelEncoder()
        df[feature] = le.fit_transform(df[feature])
        encoder_dict[feature] = le

    # remove those with timestamp

    # Finding the colmuns having NaN values
    nan_features = df.columns[df.isna().any()].tolist()
    print("Features with NaN values:", nan_features)

    # Dropping any sample containing NaN or missing value
    df = df.dropna()

    # Replacing the infinite values with a very large number
    for column in df.columns:
        if (df[column] == np.inf).any():
            df[column].replace([np.inf], [np.finfo('float32').max], inplace=True)

    return df

def split_into_train_test(df, column_name="Label"):
    # Split the data into features (X) and labels (y)
    X = df.drop(column_name, axis=1)
    y = df[column_name]

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    return X_train, X_test, y_train, y_test


def get_cicids_train_test():
    df = get_final_df()
    df = clean_df(df)
    X_train, X_test, y_train, y_test = split_into_train_test(df)
    return X_train, X_test, y_train, y_test


def get_ton_iot_train_test():
    df = pd.read_csv(TON_IOT_ROOT + "train_test_network.csv", encoding="utf-8-sig", low_memory=False)
    numerical_features = df.select_dtypes(include=[np.number]).columns.tolist()
    df.replace("-", np.nan, inplace=True)
    # Clean the data
    df = clean_data(df, encode_features=iot_ton_encode_features)
    X_train, X_test, y_train, y_test = split_into_train_test(df, column_name="label")
    return X_train, X_test, y_train, y_test


def get_usnw_train_test():
    train_df = pd.read_csv(USNW_ROOT + "UNSW_NB15_training-set.csv")
    test_df = pd.read_csv(USNW_ROOT + "UNSW_NB15_testing-set.csv")
    X_train, X_test, y_train, y_test = preprocess_usnw_data(train_df, test_df)
    return X_train, X_test, y_train, y_test


def get_nsl_kdd_train_test():
    train_df = pd.read_csv(NSL_KDD_ROOT + "KDDTrain+.txt", header=None, names=range(42))
    test_df = pd.read_csv(NSL_KDD_ROOT + "KDDTest+.txt", header=None, names=range(42))
    test_df = clean_data(test_df, nsl_kdd_encode_features)
    train_df = clean_data(train_df, nsl_kdd_encode_features)

    X_train = train_df.drop(41, axis=1)
    y_train = train_df[41]

    X_test = test_df.drop(41, axis=1)
    y_test = test_df[41]

    return X_train, X_test, y_train, y_test


def clean_data(df, encode_features):
    encoder_dict = {}
    for feature in encode_features:
        le = LabelEncoder()
        df[feature] = le.fit_transform(df[feature])
        encoder_dict[feature] = le

    # Check for NaN values and remove rows with any NaNs
    nan_features = df.columns[df.isna().any()].tolist()
    if nan_features:
        print("Features with NaN values:", nan_features)
    df = df.dropna()

    # Replace infinite values (both positive and negative)
    df.replace([np.inf, -np.inf], np.finfo('float32').max, inplace=True)

    return df



def get_cat_columns(df):
    """Return a list of categorical columns in the DataFrame."""
    return [col for col in df.columns if df[col].dtype == 'object']

def feature_process(df):
    """Process categorical features."""
    df.loc[~df['state'].isin(['FIN', 'INT', 'CON', 'REQ', 'RST']), 'state'] = 'others'
    df.loc[~df['service'].isin(['-', 'dns', 'http', 'smtp', 'ftp-data', 'ftp', 'ssh', 'pop3']), 'service'] = 'others'
    df.loc[df['proto'].isin(['igmp', 'icmp', 'rtp']), 'proto'] = 'igmp_icmp_rtp'
    df.loc[~df['proto'].isin(['tcp', 'udp', 'arp', 'ospf', 'igmp_icmp_rtp']), 'proto'] = 'others'
    return df

def preprocess_usnw_data(train_df, test_df):
    # Pre-processing: Remove specified columns
    drop_columns = ['attack_cat', 'id']
    train_df.drop(drop_columns, axis=1, inplace=True)
    test_df.drop(drop_columns, axis=1, inplace=True)

    # Separate features and labels
    X_train, y_train = train_df.drop(['label'], axis=1), train_df['label']
    X_test, y_test = test_df.drop(['label'], axis=1), test_df['label']

    # Feature pre-process
    X_train = feature_process(X_train)
    X_test = feature_process(X_test)

    # Identify categorical and non-categorical columns
    categorical_columns = get_cat_columns(X_train)
    non_categorical_columns = [col for col in X_train.columns if col not in categorical_columns]

    # Apply label encoding or one-hot encoding
    label_encoders = {}
    for column in categorical_columns:
        le = LabelEncoder()
        # Fit on the training data
        X_train[column] = le.fit_transform(X_train[column])
        X_test[column] = le.transform(X_test[column])  # Apply the same transformation to test data
        label_encoders[column] = le  # Save the encoder if needed for inverse transform

    # Scaling non-categorical features
    scaler = StandardScaler()
    X_train[non_categorical_columns] = scaler.fit_transform(X_train[non_categorical_columns])
    X_test[non_categorical_columns] = scaler.transform(X_test[non_categorical_columns])
    return X_train, X_test, y_train, y_test

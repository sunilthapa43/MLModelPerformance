from setuptools.command.build import build
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, GRU, Dense, Input


# Define models
def initialize_models(X_train, X_test):
    models = {
        'AdaBoost': AdaBoostClassifier(
            estimator=DecisionTreeClassifier(max_depth=3, class_weight='balanced'),
            n_estimators=300,
            random_state=42,
            learning_rate=0.5
        ),
        'DecisionTree': DecisionTreeClassifier(
            max_depth=10,  # Adjust this based on your dataset and experimentation
            class_weight='balanced',  # Handles imbalanced classes
            criterion='gini',  # Use "gini" or "entropy" based on experiments
            random_state=42  # Ensures reproducibility
        ),
        'GradientBoosting': GradientBoostingClassifier(
            n_estimators=100,
            max_depth=10,
            learning_rate=0.4,  # Handles class imbalance
            random_state=42
        ),
        'MLP': build_mlp_model(X_train, X_test),
        'LSTM': build_lstm_model(X_train, X_test),
        'GRU': build_gru_model(X_train, X_test)
    }
    print("Models Initialization complete")
    return models


# Define LSTM model
def build_lstm_model(X_train, X_test):
    X_train, X_test = reshape_df(X_train, X_test)
    model = Sequential()
    model.add(LSTM(64, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2]), activation='tanh'))
    model.add(LSTM(32, activation='tanh'))
    model.add(Dense(16, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return {"model":model, "X_train":X_train, "X_test":X_test}


# Define GRU model
def build_gru_model(X_train, X_test):
    X_train, X_test = reshape_df(X_train, X_test)
    model = Sequential()
    model.add(GRU(64, input_shape=(X_train.shape[1], X_train.shape[2]), activation='tanh'))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return {"model":model, "X_train":X_train, "X_test":X_test}


def build_mlp_model(X_train, X_test):
    model = Sequential([
        Input(shape=(X_train.shape[1],)),  # Define the input shape here
        Dense(64, activation='relu'),
        Dense(32, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model


def reshape_df(df1, df2):
    df = df1.values.reshape((df1.shape[0], 1, df1.shape[1]))  # (samples, timesteps, features)
    df_a = df2.values.reshape((df2.shape[0], 1, df2.shape[1]))
    return df, df_a
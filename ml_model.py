from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, GRU, Dense, Input


# Define models
def initialize_models(input_shape):
    models = {
        'AdaBoost': AdaBoostClassifier(
            estimator=DecisionTreeClassifier(max_depth=3, class_weight='balanced'),
            n_estimators=300,
            random_state=42,
            learning_rate=0.5
        ),
        # 'DecisionTree': DecisionTreeClassifier(),
        # 'GradientBoosting': GradientBoostingClassifier(n_estimators=100),
        'MLP': build_mlp_model(input_shape),
        # 'LSTM': build_lstm_model(input_shape),
        # 'GRU': build_gru_model(input_shape)
    }
    return models


# Define LSTM model
def build_lstm_model(input_shape):
    model = Sequential()
    model.add(LSTM(64, input_shape=input_shape))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model


# Define GRU model
def build_gru_model(input_shape):
    model = Sequential()
    model.add(GRU(64, input_shape=input_shape))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model


def build_mlp_model(input_shape):
    model = Sequential([
        Input(shape=(input_shape,)),  # Define the input shape here
        Dense(64, activation='relu'),
        Dense(32, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

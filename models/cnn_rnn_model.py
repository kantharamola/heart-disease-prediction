import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.models import Sequential

def build_model(input_shape):
    model = Sequential()
    
    # CNN Layers
    model.add(layers.Conv1D(64, kernel_size=3, input_shape=input_shape, padding="same"))
    model.add(layers.BatchNormalization())  # Normalization
    model.add(layers.LeakyReLU())  # Better than ReLU
    model.add(layers.MaxPooling1D(pool_size=2))

    model.add(layers.Conv1D(128, kernel_size=3, padding="same"))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())
    model.add(layers.MaxPooling1D(pool_size=2))
    model.add(layers.Dropout(0.3))  # Reduce overfitting

    # RNN Layers
    model.add(layers.Bidirectional(layers.LSTM(64, return_sequences=True, dropout=0.3)))
    model.add(layers.Bidirectional(layers.LSTM(32)))

    # Fully Connected Layer
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dropout(0.3))
    model.add(layers.Dense(1, activation='sigmoid'))  # Output Layer

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

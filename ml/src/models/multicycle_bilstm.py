"""
Stacked Bidirectional LSTM for multi-cycle (windowed) capacity prediction.

Input shape : (window_size, n_features)
Output      : scalar capacity (scaled — call the y-scaler externally)

Architecture (matches the proof-of-concept training notebook):
    BiLSTM(128, return_sequences=True) → Dropout(0.2)
    BiLSTM(64,  return_sequences=True) → Dropout(0.2)
    LSTM(32)                            → Dense(1)
"""

from __future__ import annotations

from typing import Tuple


def build_multicycle_bilstm(
    window_size: int,
    n_features: int,
    learning_rate: float = 1e-4,
    units: Tuple[int, int, int] = (128, 64, 32),
    dropout: float = 0.2,
):
    import tensorflow as tf
    from tensorflow.keras.layers import LSTM, Bidirectional, Dense, Dropout
    from tensorflow.keras.models import Sequential

    u1, u2, u3 = units
    model = Sequential(
        [
            Bidirectional(
                LSTM(u1, activation="tanh", return_sequences=True),
                input_shape=(window_size, n_features),
            ),
            Dropout(dropout),
            Bidirectional(LSTM(u2, activation="relu", return_sequences=True)),
            Dropout(dropout),
            LSTM(u3, activation="relu"),
            Dense(1),
        ]
    )
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss="mse",
    )
    return model

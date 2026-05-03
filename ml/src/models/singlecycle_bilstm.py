"""
Stacked Bidirectional LSTM that consumes a single discharge curve directly.

Input shape : (n_timesteps, n_channels)   — e.g. (1000, 3) for V/I/T
Output      : scalar capacity (scaled — call the y-scaler externally)

Architecture mirrors the proof-of-concept single-cycle notebook.
"""

from __future__ import annotations

from typing import Tuple


def build_singlecycle_bilstm(
    n_timesteps: int,
    n_channels: int,
    learning_rate: float = 3e-4,
    units: Tuple[int, int, int] = (128, 64, 32),
    dropouts: Tuple[float, float, float] = (0.3, 0.3, 0.2),
    head_units: int = 32,
):
    import tensorflow as tf
    from tensorflow.keras.layers import (
        LSTM,
        Bidirectional,
        Dense,
        Dropout,
        Input,
    )
    from tensorflow.keras.models import Model

    u1, u2, u3 = units
    d1, d2, d3 = dropouts

    inp = Input(shape=(n_timesteps, n_channels), name="discharge_curve_input")
    x = Bidirectional(LSTM(u1, return_sequences=True))(inp)
    x = Dropout(d1)(x)
    x = Bidirectional(LSTM(u2, return_sequences=True))(x)
    x = Dropout(d2)(x)
    x = Bidirectional(LSTM(u3))(x)
    x = Dropout(d3)(x)
    x = Dense(head_units, activation="relu")(x)
    out = Dense(1, activation="linear", name="capacity_output")(x)

    model = Model(inp, out, name="Stacked_BiLSTM_SingleCycle")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss="mse",
        metrics=["mae"],
    )
    return model

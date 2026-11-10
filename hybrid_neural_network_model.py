import tensorflow as tf
from tensorflow.keras.metrics import Metric
from tensorflow.keras import Input
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (Dense, Bidirectional, LSTM, Conv1D, Dropout,
                                     BatchNormalization, Concatenate, Flatten,Add)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2


# Custom focal Loss =============================================================================
def focal_loss(gamma=2.5, alpha=0.75):
    def loss(y_true, y_pred):
        epsilon = tf.keras.backend.epsilon()
        y_pred = tf.clip_by_value(y_pred, epsilon, 1. - epsilon)
        pt = y_true * y_pred + (1 - y_true) * (1 - y_pred)
        return -alpha * (1 - pt) ** gamma * tf.math.log(pt)
    return loss

# F1-Score metric ===============================================================================
class F1Score(Metric):
    def __init__(self, threshold=0.5, name='f1', **kwargs):
        super().__init__(name=name, **kwargs)
        self.threshold = threshold
        self.precision = tf.keras.metrics.Precision(thresholds=threshold)
        self.recall = tf.keras.metrics.Recall(thresholds=threshold)

    def update_state(self, y_true, y_pred, sample_weight=None):
        self.precision.update_state(y_true, y_pred, sample_weight)
        self.recall.update_state(y_true, y_pred, sample_weight)

    def result(self):
        p = self.precision.result()
        r = self.recall.result()
        return 2 * ((p * r) / (p + r + tf.keras.backend.epsilon()))

    def reset_state(self):
        self.precision.reset_state()
        self.recall.reset_state()

# 3. HybridAttention1D (Position+Channel) Attention ====================================================
class HybridAttention1D(tf.keras.layers.Layer):
    def __init__(self, reduction=8, dropout_rate=0.4, l2_reg=1e-4, **kwargs):  # Tuned defaults
        super().__init__(**kwargs)
        self.reduction = reduction
        self.dropout_rate = dropout_rate
        self.l2_reg = l2_reg

    def build(self, input_shape):
        _, self.T, self.C = input_shape
        reduced_channels = max(self.C // self.reduction, 4)

        # Channel Attention path
        self.channel_fc1 = Dense(reduced_channels, activation='relu', kernel_regularizer=l2(self.l2_reg))
        self.channel_fc2 = Dense(self.C, activation='sigmoid', kernel_regularizer=l2(self.l2_reg))
        self.channel_dropout = Dropout(self.dropout_rate)

        # Position Attention path
        self.position_fc1 = Dense(reduced_channels, activation='relu', kernel_regularizer=l2(self.l2_reg))
        self.position_fc2 = Dense(self.T, activation='sigmoid', kernel_regularizer=l2(self.l2_reg))
        self.position_dropout = Dropout(self.dropout_rate)
        super().build(input_shape)

    def call(self, x):
        # 1. Global Average Pooling for Channels
        ch_desc = tf.reduce_mean(x, axis=1)
        ch_attn = self.channel_fc1(ch_desc)
        ch_attn = self.channel_dropout(ch_attn)
        ch_attn = self.channel_fc2(ch_attn)
        ch_attn = tf.expand_dims(ch_attn, axis=1)

        # 2. Global Average Pooling for Positions
        pos_desc = tf.reduce_mean(x, axis=2)
        pos_attn = self.position_fc1(pos_desc)
        pos_attn = self.position_dropout(pos_attn)
        pos_attn = self.position_fc2(pos_attn)
        pos_attn = tf.expand_dims(pos_attn, axis=-1) #

        # 3. Feature Recalibration
        return x * ch_attn * pos_attn

# 4. HybridModel Building =============================================================================
class hybridmodel:
    @staticmethod
    def build(input_shape, lr=None):
        inputs = Input(shape=input_shape)

        # 1. CNN branch (64 units)
        cnn = Conv1D(64, kernel_size=3, padding="same", activation="relu", kernel_regularizer=l2(1e-4))(inputs)
        cnn = BatchNormalization()(cnn)
        cnn = Dropout(0.4)(cnn)

        # 2. BiLSTM branch (64 units)
        bilstm = Bidirectional(LSTM(64, return_sequences=True, kernel_regularizer=l2(1e-4)))(inputs)
        bilstm = Dropout(0.4)(bilstm)

        # 3. Concatenate CNN and BiLSTM branches
        # This is our "base" representation of the data before attention
        merged_features = Concatenate(axis=-1)([cnn, bilstm])

        # 4. HybridAttention Module
        # We apply attention to the merged features
        attention_output = HybridAttention1D(reduction=8, dropout_rate=0.4)(merged_features)

        # ---> THE RESIDUAL CONNECTION <---
        # We add the pre-attention features to the post-attention features.
        x = Add()([merged_features, attention_output])
        # # Normalization after the residual connection to stabilize variance
        x = BatchNormalization()(x)

        # 5. Classification Head
        x = Flatten()(x)
        x = Dense(128, activation="relu", kernel_regularizer=l2(1e-4))(x)
        x = Dropout(0.4)(x)

        # 6. Output Module
        outputs = Dense(1, activation="sigmoid")(x)
        model = Model(inputs=inputs, outputs=outputs)
        # Compile model
        model.compile(optimizer=Adam(learning_rate=lr, weight_decay=1e-6),loss=focal_loss(gamma=2.5, alpha=0.77),
                      metrics=["accuracy", tf.keras.metrics.AUC(name="auc"), F1Score()])

        return model
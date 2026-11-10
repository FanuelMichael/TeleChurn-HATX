import gc
from tensorflow.keras import backend as K
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import (accuracy_score, precision_score, recall_score,f1_score, roc_auc_score, confusion_matrix)
import tensorflow as tf
from tensorflow.keras.regularizers import l2
from tensorflow.keras import Model, Input
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (Dense, Dropout, Conv1D, MaxPooling1D, BatchNormalization,
                                     LSTM, Bidirectional, GlobalAveragePooling1D)
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
# Import custom F1Score metric class from Hybrid model module
from hybrid_neural_network_model import F1Score
from model_evaluation import render_classification_report_table


# =====================================================================================================

# Used for computing baseline models(CNN and BiLSTM) metrics inside this module---------
def _compute_metrics(y_test, y_proba, y_pred):
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    try:
        auc = roc_auc_score(y_test, y_proba)
    except ValueError:
        auc = np.nan
    return {"accuracy": round(acc, 4) * 100,"precision": round(prec, 4) * 100,"recall": round(rec, 4) * 100,
            "f1": round(f1, 4) * 100, "auc": round(auc, 4)}

# Standalone CNN model building------------------------------------
def build_cnn(input_dim):
    model = Sequential([Input(shape=(input_dim, 1)),Conv1D(filters=64, kernel_size=3, activation='relu'),
        BatchNormalization(),MaxPooling1D(pool_size=2), Dropout(0.2),Conv1D(filters=128, kernel_size=3, activation='relu'),
        BatchNormalization(),GlobalAveragePooling1D(),Dropout(0.3),Dense(1, activation='sigmoid')])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=["accuracy", tf.keras.metrics.AUC(name="auc"), F1Score()])
    return model

# Standalone BiLSTM model building------------------------------------
def build_bilstm(input_dim):
    model = Sequential([Input(shape=(input_dim, 1)),Bidirectional(LSTM(64, return_sequences=True, kernel_regularizer=l2(0.001))),
        Dropout(0.3),Bidirectional(LSTM(32, kernel_regularizer=l2(0.001))),Dropout(0.3),Dense(1, activation='sigmoid')])

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=["accuracy", tf.keras.metrics.AUC(name="auc"), F1Score()])
    return model

# Select the threshold that maximizes validation F1 --------------------------------------
def select_threshold_on_validation(y_true,y_probability,thresholds=None):

    if thresholds is None:
        thresholds = np.arange(0.10, 0.90, 0.01)
    y_true = np.asarray(y_true).astype(int)
    y_probability = np.asarray(y_probability).ravel()
    best_threshold = 0.50
    best_f1 = -1.0

    for threshold in thresholds:
        predictions = (y_probability >= threshold).astype(int)
        score = f1_score(y_true,predictions,zero_division=0)
        if score > best_f1:
            best_f1 = score
            best_threshold = float(threshold)
    return best_threshold, best_f1

# Main comparison function to match hybrid validation strategy
def compare_models(X_train_res, y_train_res, X_val, y_val, X_test, y_test,
                   hybrid_metrics=None, epochs=30, batch_size=64):

    Comparison_metrics_results = {}
    input_dim = X_train_res.shape[1]

    # 1. Reshape X_train_res: (samples, features, 1)
    if len(X_train_res.shape) == 2:
        X_train_seq = X_train_res.reshape(-1, input_dim, 1)
    else:
        X_train_seq = X_train_res.reshape(X_train_res.shape[0], X_train_res.shape[1], 1)

    # 2. Reshape X_val: (samples, features, 1)
    if len(X_val.shape) == 2:
        X_val_seq = X_val.reshape(-1, input_dim, 1)
    else:
        X_val_seq = X_val.reshape(X_val.shape[0], X_val.shape[1], 1)

    # 3. Reshape X_test: (samples, features, 1)
    if len(X_test.shape) == 2:
        X_test_seq = X_test.reshape(-1, input_dim, 1)
    else:
        X_test_seq = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

    # CNN Training, Validation Threshold Selection and Test Evaluation -------------------
    print("# === CNN Training, Validation Threshold Selection and Test Evaluation ===")
    cnn_callbacks = [EarlyStopping(monitor="val_f1",mode="max",patience=6, restore_best_weights=True),
                     ReduceLROnPlateau( monitor="val_f1",mode="max", factor=0.5, patience=3,min_lr=1e-6,verbose=1)]
    cnn = build_cnn(input_dim)
    cnn.fit(X_train_seq,y_train_res,epochs=epochs, batch_size=batch_size,validation_data=(X_val_seq, y_val),
        verbose=0,callbacks=cnn_callbacks)

    # Threshold selection uses VALIDATION predictions only.
    proba_cnn_val = cnn.predict(X_val_seq, verbose=0).ravel()
    cnn_threshold, cnn_val_f1 = select_threshold_on_validation(y_val,proba_cnn_val)
    # Final CNN predictions on the untouched test set.
    proba_cnn = cnn.predict(X_test_seq, verbose=0).ravel()
    pred_cnn = (proba_cnn >= cnn_threshold).astype(int)
    print(f"CNN validation threshold: {cnn_threshold:.2f} | "f"Validation F1: {cnn_val_f1:.4f}")
    Comparison_metrics_results["CNN"] = _compute_metrics( y_test, proba_cnn, pred_cnn)
    cnn_eval_fig = _plot_baseline_evaluation(y_test, pred_cnn, "CNN")

    # --- BiLSTM Training, Validation Threshold Selection and Test Evaluation ---
    print("# === BiLSTM Training, Validation Threshold Selection and Test Evaluation ===")
    bilstm_callbacks = [EarlyStopping(monitor="val_f1",mode="max",patience=6,restore_best_weights=True),
                       ReduceLROnPlateau( monitor="val_f1",mode="max",factor=0.5,patience=3,min_lr=1e-6,verbose=1)]
    bilstm = build_bilstm(input_dim)
    bilstm.fit( X_train_seq,y_train_res, epochs=epochs, batch_size=batch_size, validation_data=(X_val_seq, y_val),
        verbose=0,callbacks=bilstm_callbacks)

    # Threshold selection uses BiLSTM validation predictions only.
    proba_bilstm_val = bilstm.predict(X_val_seq, verbose=0).ravel()
    bilstm_threshold, bilstm_val_f1 = select_threshold_on_validation(y_val,proba_bilstm_val)

    # Final BiLSTM predictions on the untouched test set.
    proba_bilstm = bilstm.predict(X_test_seq, verbose=0).ravel()
    pred_bilstm = (proba_bilstm >= bilstm_threshold).astype(int)
    print(f"BiLSTM validation threshold: {bilstm_threshold:.2f} | "f"Validation F1: {bilstm_val_f1:.4f}")
    Comparison_metrics_results["BiLSTM"] = _compute_metrics(y_test, proba_bilstm, pred_bilstm)
    bilstm_eval_fig = _plot_baseline_evaluation(y_test, pred_bilstm, "BiLSTM")

    # Hybrid Deep Learning Model
    if hybrid_metrics:
        Comparison_metrics_results["1D CNN-BiLSTM-HybridAttention"] = {
            "accuracy": hybrid_metrics["accuracy"], "precision": hybrid_metrics["precision"],
            "recall": hybrid_metrics["recall"], "f1": hybrid_metrics["f1"],
            "auc": hybrid_metrics.get("roc_auc", np.nan)}

        # Convert to DataFrame
        results_df = pd.DataFrame(Comparison_metrics_results).T[["accuracy", "precision", "recall", "f1", "auc"]]
        results_df = results_df.round(2)

        # Visualization
        fig = plt.figure(figsize=(20, 10), dpi=300)
        ax_table = plt.subplot2grid((2, 1), (0, 0))
        ax_table.axis("off")
        ax_table.set_title("Models Performance Comparison",
                           fontsize=15, pad=25, fontweight='bold', color='#0F172A')

        col_labels = ["Model", "Accuracy", "Precision", "Recall", "F1-Score", "AUC"]
        table_data = []
        for model in results_df.index:
            row = [model] + [f"{results_df.loc[model, col]:.2f}%" if col != "auc"
                else f"{results_df.loc[model, col]:.4g}"
                for col in ["accuracy", "precision", "recall", "f1", "auc"]]
            table_data.append(row)

        table = ax_table.table( cellText=table_data,colLabels=col_labels,loc='center',cellLoc='center',bbox=[0, 0.1, 1, 0.8])
        table.auto_set_font_size(False)
        table.set_fontsize(11)
        for (i, j), cell in table.get_celld().items():
            if i == 0:
                cell.set_facecolor('#1E293B')
                cell.set_text_props(weight='bold', color='white', fontsize=12)
                cell.set_edgecolor('#1E293B')
            else:
                model_name = table_data[i - 1][0]
                if "HybridAttention" in model_name or "Hybrid" in model_name:
                    cell.set_facecolor('#EFF6FF')
                    cell.set_edgecolor('#BFDBFE')
                    if j == 0:
                        cell.set_text_props(weight='bold', color='#1E40AF', fontsize=11)
                    else:
                        cell.set_text_props(weight='bold', color='#1E40AF', fontsize=11)
                else:
                    cell.set_facecolor('#FFFFFF' if i % 2 == 0 else '#F8FAFC')
                    cell.set_edgecolor('#E2E8F0')
                    cell.set_text_props(color='#334155', fontsize=10.5)
                    if j == 0:
                        cell.set_text_props(weight='bold', color='#0F172A')

        plt.subplots_adjust(left=0.06, right=0.94, top=0.85, bottom=0.1, hspace=0.06)
        plt.close(fig)
        K.clear_session(); gc.collect()
        return cnn_eval_fig, bilstm_eval_fig, fig

# =====Plot standalone baseline (cnn and bilstm) evaluation=========================================
def _plot_baseline_evaluation(y_test, y_pred, model_name):
    fig = plt.figure(figsize=(16, 7), dpi=200)

    # --- 1. Confusion Matrix ---
    ax_cm = plt.subplot(1, 2, 1)
    cm = confusion_matrix(y_test, y_pred)
    im = ax_cm.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    ax_cm.figure.colorbar(im, ax=ax_cm)

    classes = ['0', '1']
    tick_marks = np.arange(len(classes))
    ax_cm.set_xticks(tick_marks)
    ax_cm.set_xticklabels(classes, rotation=45)
    ax_cm.set_yticks(tick_marks)
    ax_cm.set_yticklabels(classes)

    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax_cm.text(j, i, format(cm[i, j], 'd'), ha="center", va="center",
                       color="white" if cm[i, j] > thresh else "black")

    ax_cm.set_title(f"Confusion Matrix - {model_name}", fontsize=14, fontweight='bold')
    ax_cm.set_ylabel('True')
    ax_cm.set_xlabel('Predicted')

    # --- 2. Classification Report (Using Reusable Helper) ---
    ax_table = plt.subplot(1, 2, 2)
    render_classification_report_table(y_test=y_test,y_pred=y_pred,ax=ax_table,
        title=f"Classification Report - {model_name}",scale=(1.2, 1.6),header_color='#1E293B')

    plt.tight_layout()
    plt.close(fig)
    return fig
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style='whitegrid')
from sklearn.metrics import (confusion_matrix, classification_report, roc_curve,
                             roc_auc_score, precision_score, recall_score, f1_score)

#==============================================================================================
# Class Distribution Visualisation before and after applying SMOTE  -----------
def plot_class_distribution(before_counts, after_counts):
    # Calculate percentages
    before_total = sum(before_counts.values())
    after_total = sum(after_counts.values())
    before_percent = {k: v / before_total * 100 for k, v in before_counts.items()}
    after_percent = {k: v / after_total * 100 for k, v in after_counts.items()}

    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    bar_colors = ['blue', 'red']
    # Before SMOTE resampling
    axes[0].bar(before_counts.keys(), before_counts.values(), color=bar_colors)
    axes[0].set_title("Class Distribution Before Resampling")
    axes[0].set_ylabel("Sample Count")
    for i, (label, count) in enumerate(before_counts.items()):
        pct = before_percent[label]
        axes[0].text(i, count + 50, f"{count} ({pct:.1f}%)", ha='center', fontsize=10)
    # After SMOTE resampling
    axes[1].bar(after_counts.keys(), after_counts.values(), color=bar_colors)
    axes[1].set_title("Class Distribution After SMOTE")
    for i, (label, count) in enumerate(after_counts.items()):
        pct = after_percent[label]
        axes[1].text(i, count + 50, f"{count} ({pct:.1f}%)", ha='center', fontsize=10)
    plt.tight_layout()
    plt.close(fig)
    return fig

# Performance metrics, confusion matrix and AUC curve of the hybrid model. -----------------------------
def plot_evaluation(y_test, y_proba, y_pred):
    fig = plt.figure(figsize=(18, 14))

    # 1. Confusion Matrix Subplot
    ax_conf = plt.subplot(2, 2, 1)
    conf_matrix = confusion_matrix(y_test, y_pred)
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', ax=ax_conf)
    ax_conf.set_title('Confusion Matrix')
    ax_conf.set_xlabel('Predicted')
    ax_conf.set_ylabel('True')

    # 2. ROC-AUC Subplot
    ax_roc = plt.subplot(2, 2, 2)
    fpr, tpr, thresholds = roc_curve(y_test, y_proba)
    roc_auc = roc_auc_score(y_test, y_proba)

    # Calculate optimal threshold
    optimal_idx = np.argmax(tpr - fpr)
    optimal_threshold = thresholds[optimal_idx]
    ax_roc.plot(fpr, tpr, color='darkorange', lw=2, label=f'AUC = {roc_auc:.3f}')
    ax_roc.plot([0, 1], [0, 1], color='navy', lw=1, linestyle='--')
    ax_roc.scatter(fpr[optimal_idx], tpr[optimal_idx], color='red',
                   label=f'Best Threshold: {optimal_threshold:.2f}')
    ax_roc.set_title('ROC Curve')
    ax_roc.set_xlabel('FPR')
    ax_roc.set_ylabel('TPR')
    ax_roc.legend(loc='lower right')
    # 3. Detailed Metrics Table (Using the Reusable Helper)
    ax_table = plt.subplot(2, 1, 2)
    render_classification_report_table(y_test=y_test,y_pred=y_pred, ax=ax_table,
                                       title="Detailed Metrics",scale=(0.5, 1.6),header_color='#1E293B')
    plt.subplots_adjust(top=0.90, bottom=0.02, hspace=0.1)
    plt.close(fig)
    return fig

# returns detailed metrics(classification_report) of model.----------------------------
def detailed_metrics(y_test, y_pred, y_proba):
    report = classification_report(y_test, y_pred, output_dict=True)
    acc = (y_pred == y_test).mean()
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    roc_auc = roc_auc_score(y_test, y_proba)
    return {'classification_report': report,'accuracy': round(float(acc),4),'precision': round(float(prec),4),
        'recall': round(float(rec),4),'f1': round(float(f1),4),'roc_auc': round(float(roc_auc),4)}


# Function to generate and render a styled classification report table.-----------------------------
def render_classification_report_table(y_test, y_pred, ax, title="Detailed Metrics", scale=(0.5, 1.6),
                                       header_color=None):
    ax.axis('off')
    # 1. Generate Data
    report = classification_report(y_test, y_pred, output_dict=True)
    report_df = pd.DataFrame(report).transpose()
    report_df.index.name = "Class"
    report_df.reset_index(inplace=True)

    # 2. Clean Data
    class_col = report_df.columns[0]
    report_df[class_col] = report_df[class_col].replace({
        'accuracy': 'Accuracy', 'macro avg': 'Macro avg', 'weighted avg': 'Weighted avg'})

    numeric_cols = ['precision', 'recall', 'f1-score', 'support']
    report_df[numeric_cols] = report_df[numeric_cols].round(4)

    # 3. Render Table
    table_data = report_df.values.tolist()
    column_labels = ['Class', 'Precision', 'Recall', 'F1-Score', 'Support']
    table = ax.table(cellText=table_data, colLabels=column_labels, loc='center', cellLoc='center')
    table.scale(*scale)
    table.auto_set_font_size(False)

    # 4. Apply Styling
    for col_idx, _ in enumerate(column_labels):
        cell = table[(0, col_idx)]
        cell.set_text_props(weight='bold', color='white' if header_color else 'black')
        if header_color:
            cell.set_facecolor(header_color)

    # Bold specific rows
    target_rows = ['Accuracy', 'Macro avg', 'Weighted avg']
    for row_idx in range(1, len(table_data) + 1):
        first_cell = table[(row_idx, 0)]
        if first_cell.get_text().get_text() in target_rows:
            first_cell.set_text_props(weight='bold')

    ax.set_title(title, fontsize=14, fontweight='bold')
    return table

# Function for plotting the final feature columns after cleaning, OHE, and imputation.--------------------------
def plot_preprocessed_features(feature_names):

    fig, ax = plt.subplots(figsize=(16, 6))
    # Dummy values (height of 1) to visualize feature presence
    y_vals = np.ones(len(feature_names))
    # Create vertical bars
    ax.bar(range(len(feature_names)), y_vals, color='#3b82f6', edgecolor='black')
    # Configure X-axis
    ax.set_xticks(range(len(feature_names)))
    ax.set_xticklabels(feature_names, rotation=45, ha='right', fontsize=13, fontweight='bold')
    # Hide Y-axis ticks (since we are just showing presence)
    ax.set_yticks([])
    ax.set_ylabel("Feature Presence", fontsize=14, fontweight='bold')
    # Styling
    ax.set_title(f"Final Preprocessed Feature Set ({len(feature_names)} Columns)",
                 fontsize=16, fontweight='bold', pad=15)
    # Remove top, right, and left spines for a cleaner look
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    plt.tight_layout()
    plt.close(fig)

    return fig

# Generates a table plot for the Anchor Rules validation results.----------------------------
def plot_anchor_rules_table(shap_res, pdp_res, dale_res):
    fig, ax = plt.subplots(figsize=(14, 4))
    ax.axis('off')
    # Column headers
    col_labels = ['Rule Type', 'Logical Condition (Anchor)', 'Coverage', 'Precision']
    # Table data (using logical AND symbol ∧)
    data = [['SHAP(A_SHAP_Risk)', '{tenure ≤ -0.96 AND Contract_Month-to-month = 1 AND MonthlyCharges > 0.83}',
            f"{shap_res[0] * 100:.2f}%", f"{shap_res[1] * 100:.2f}%"],
            ['PDP(A_PDP_Inflection)', '{tenure ≤ -0.50 AND MonthlyCharges > 0.50}',
            f"{pdp_res[0] * 100:.2f}%", f"{pdp_res[1] * 100:.2f}%"],
            ['DALE(A_DALE_Cliff)', '{tenure ≤ -1.00 AND MonthlyCharges ≥ 1.00}',
            f"{dale_res[0] * 100:.2f}%", f"{dale_res[1] * 100:.2f}%"]]
    # Create table
    table = ax.table(cellText=data, colLabels=col_labels,colWidths=[0.18, 0.52, 0.15, 0.15], loc='center', cellLoc='center')
    # Styling the table
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1, 2.5)
    for (row, col), cell in table.get_celld().items():
        cell.set_edgecolor('#CBD5E1')  # Light gray borders
        if row == 0:
            # Header styling
            cell.set_text_props(weight='bold', color='white')
            cell.set_facecolor('#1E293B')  # Dark header
            cell.set_height(0.12)
        else:
            # Row styling (Zebra striping)
            if row % 2 == 0:
                cell.set_facecolor('#F8FAFC')  # Soft cool gray
            else:
                cell.set_facecolor('#FFFFFF')  # Pure white
            cell.set_text_props(color='#334155')
            # Make Rule Type bold
            if col == 0:
                cell.set_text_props(weight='bold', color='#0F172A')
    ax.set_title("Empirical Validation of Triangulated Anchor Rules",fontsize=12, fontweight='bold', pad=2, color='#0F172A')
    plt.tight_layout()
    plt.close(fig)
    return fig

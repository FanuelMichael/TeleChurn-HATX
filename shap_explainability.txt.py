import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
import shap
import numpy as np
import logging
logging.getLogger('matplotlib.font_manager').setLevel(logging.ERROR)
import matplotlib.pyplot as plt
rng = np.random.default_rng(42)


#================================================================================================================
class shap_explainer:
    def __init__(self, model, feature_names, X_train, X_test,background_samples=50):
        self.model = model
        self.feature_names = feature_names

        # Convert 3D → 2D
        self.X_train_2d = self._to_2d(X_train)
        self.X_test_2d = self._to_2d(X_test)

        # Subsample background for Gradient explainer
        n_background = min(background_samples, len(self.X_train_2d))
        background_idx = np.random.choice(len(self.X_train_2d), n_background, replace=False)
        self.X_background = self.X_train_2d[background_idx]

        # Set explainer to faster shap gradient explainer
        self.explainer = shap.GradientExplainer(self.model,self.X_background,batch_size=256)

        # Compute SHAP values only for the test subset
        self.shap_exp = self.explainer(self.X_test_2d)
        self.shap_vals = self.shap_exp.values
        self.expected_value = self.shap_exp.base_values

        # GradientExplainer sometimes returns a list (for multi-class) or 3D array
        if isinstance(self.shap_vals, list):
            self.shap_vals = self.shap_vals[0]  # Take the first class (Churn)

        # Converts (N, F, 1) to (N, F) N= samples, F= features
        if len(self.shap_vals.shape) == 3:
            self.shap_vals = self.shap_vals.squeeze(-1)

        # --- Handle None base values ---
        if self.expected_value is None or (isinstance(self.expected_value, np.ndarray) and self.expected_value.dtype == object):
            # Calculate the mean prediction on the background data manually
            bg_preds = self.model.predict(self.X_background, verbose=0).ravel()
            self.expected_value = np.mean(bg_preds)

    # Used to convert 3D → 2D
    def _to_2d(self, arr):
        return arr.reshape(arr.shape[0], arr.shape[2]) if arr.ndim == 3 else arr

#===============================SHAP Explainability==============================================================
    # 1. SHAP Summary plot------------------------------------------------------------
    def plot_summary(self, ax=None):
        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 6))
        shap.summary_plot(self.shap_vals,self.X_test_2d,feature_names=self.feature_names,
            max_display=12,show=False,rng=rng)
        ax.set_title("SHAP Summary (Global Importance)")
        plt.tight_layout()
        fig = plt.gcf()
        plt.close(fig)
        return fig

    # 2. SHAP Interaction Plot ------------------------------------------------------
    def shap_interaction_plot(self, features=None, top_k=8):
        # 1. Select features
        if features is None:
            features = self.get_top_shap_features(top_n=top_k + 1)
        # 2. Validate features
        valid_features = [f for f in features if f in self.feature_names]
        if len(valid_features) < 2:
            print("Not enough valid features for interaction plotting.")
            return
        # 3. Build feature pairs (f1,f2), (f2,f3), ...
        pairs = []
        for i in range(len(valid_features) - 1):
            pairs.append((valid_features[i], valid_features[i + 1]))

        # Use only top_k pairs
        pairs = pairs[:top_k]
        # 4. Split into two groups of 4
        group1 = pairs[:4]
        group2 = pairs[4:8]
        # FIGURE 1 (first 4 plots)
        fig1, axes1 = plt.subplots(2, 2, figsize=(10, 8))
        axes1 = axes1.flatten()
        for idx, (f1, f2) in enumerate(group1):
            ax = axes1[idx]
            print(f"SHAP Interaction Plot: {f1} × {f2}")
            shap.dependence_plot(f1, self.shap_vals, self.X_test_2d, feature_names=self.feature_names,
                                 interaction_index=f2, ax=ax, show=False)
            ax.set_title(f"{f1} × {f2}")

        plt.suptitle("SHAP Interaction Plot", fontsize=12)
        plt.tight_layout()
        plt.close(fig1)

        # FIGURE 2 (next 4 plots)
        fig2, axes2 = plt.subplots(2, 2, figsize=(10, 8))
        axes2 = axes2.flatten()

        for idx, (f1, f2) in enumerate(group2):
            ax = axes2[idx]
            print(f"SHAP Interaction Plot: {f1} × {f2}")
            shap.dependence_plot(f1, self.shap_vals, self.X_test_2d, feature_names=self.feature_names,
                                 interaction_index=f2, ax=ax, show=False)
            ax.set_title(f"{f1} × {f2}")

        plt.suptitle("SHAP Interaction Plot", fontsize=12)
        plt.tight_layout()
        plt.close(fig2)
        return fig1, fig2

    # returns top n shap features for shap interaction plot----------------------------
    def get_top_shap_features(self, top_n=None):
        shap_abs_mean = np.mean(np.abs(self.shap_vals), axis=0)
        # Ensure we are sorting a 1D array
        if len(shap_abs_mean.shape) > 1:
            shap_abs_mean = shap_abs_mean.squeeze()
        sorted_idx = np.argsort(shap_abs_mean)[::-1].flatten()  # <-- Added .flatten()
        ordered_features = [self.feature_names[i] for i in sorted_idx]
        if top_n is not None:
            ordered_features = ordered_features[:top_n]
        return ordered_features
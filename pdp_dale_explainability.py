import numpy as np
import tensorflow as tf
import matplotlib
matplotlib.rcParams['font.family'] = 'DejaVu Sans'
matplotlib.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Helvetica', 'sans-serif']
import logging
logging.getLogger('matplotlib.font_manager').setLevel(logging.ERROR)
import matplotlib.pyplot as plt
plt.rcParams['font.weight'] = 'bold'
plt.rcParams['axes.labelweight'] = 'bold'
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_is_fitted
from sklearn.inspection import PartialDependenceDisplay

#=========================================================================================================
class _ModelWrapper(ClassifierMixin,BaseEstimator):
    _estimator_type = "classifier"
    def __init__(self, keras_model, batch_size=256):
        self.keras_model = keras_model
        self.batch_size = batch_size

    def fit(self, X, y=None):
        self.n_features_in_ = X.shape[1]
        self.classes_ = np.array([0, 1])
        # Use the exact attribute name Scikit-Learn expects for fitted validation
        self.is_fitted_ = True
        return self

    def predict_proba(self, X):
        check_is_fitted(self, attributes=['is_fitted_'])

        # Safely reshape 2D sklearn input back to 3D Keras input (samples, features, 1)
        # Scikit-learn passes 2D arrays during PDP evaluation,
        # But our keras 1DCNN-BiLSTM model strictly requires 3D input.
        if X.ndim == 2:
            X_keras = X.reshape((X.shape[0], X.shape[1], 1))
        else:
            X_keras = X

        # Utilize Keras's built-in batch processing
        p = self.keras_model.predict(X_keras, batch_size=self.batch_size, verbose=0)
        p = np.asarray(p).reshape(p.shape[0], -1)   # Flatten to (n, 1)
        p = p[:, -1]
        # Scikit-Learn PDP requires 2 columns: [Probability(Class 0), Probability(Class 1)]
        return np.column_stack([1 - p, p])

    def predict(self, X):
        proba = self.predict_proba(X)
        return (proba[:, 1] >= 0.5).astype(int)

class pdp_dale_explainer:
    def __init__(self, model, feature_names, X_train, X_test):
        self.model = model
        self.feature_names = feature_names
        self.X_train = X_train
        self.X_test = X_test

        # Convert (samples, 1, features) → (samples, features)
        self.X_train_2d = self._to_2d(X_train)
        self.X_test_2d = self._to_2d(X_test)

        # Wrap model for sklearn PDP
        self.model_wrapper = self._wrap_model()
        # Fit wrapper once with dummy labels
        dummy_y = np.random.randint(0, 2, len(self.X_train_2d))
        self.model_wrapper.fit(self.X_train_2d, dummy_y)

    # Utility: reshape 3D → 2D
    def _to_2d(self, arr):
        """
        Converts 3D Keras input (samples, features, 1) to 2D (samples, features).
        This is required because scikit-learn's PartialDependenceDisplay only accepts 2D arrays.
        """
        if arr.ndim == 3:
            if arr.shape[1] == 1:
                return arr.reshape(arr.shape[0], arr.shape[2])
            elif arr.shape[2] == 1:
                return arr.reshape(arr.shape[0], arr.shape[1])
            else:
                return arr.reshape(arr.shape[0], -1)
        return arr

    # Model Wrapper for sklearn PDP ------------------------------
    def _wrap_model(self, batch_size=256):
        return _ModelWrapper(self.model, batch_size)

    # 1. Partial Dependence Plot(PDP)--------------------------------------------------------------------
    def plot_pdp_grid(self, top_features):
        valid = [f for f in top_features if f in self.feature_names]
        if not valid:
            print("No valid features for partial dependence plot.")
            return

        n = len(valid)
        rows = int(np.ceil(n / 2))
        fig, axes = plt.subplots(rows, 2, figsize=(9, 5 * rows))
        axes = axes.flatten()

        for i, feature in enumerate(valid):
            ax = axes[i]
            idx = self.feature_names.index(feature)

            try:
                PartialDependenceDisplay.from_estimator(self.model_wrapper, self.X_test_2d, [idx],
                            feature_names=self.feature_names, ax=ax, kind='average',subsample=100)
                ax.set_title(f"PDP: {feature}")
            except Exception as e:
                print(f"PDP failed for '{feature}': {e}")
        # Hide unused axes
        for j in range(i + 1, len(axes)):
            axes[j].axis("off")
        plt.suptitle("Partial Dependence Plots", fontsize=12)
        plt.tight_layout()
        return fig

    # 2. DALE 2D visualization heatmap and DALE 3D visualization surface in one plot--------------------
    def dale_2d_3d_plot(self, feature_x, feature_y, bins=20):

        if feature_x not in self.feature_names:
            raise ValueError(f"Feature '{feature_x}' not found.")
        if feature_y not in self.feature_names:
            raise ValueError(f"Feature '{feature_y}' not found.")

        ix = self.feature_names.index(feature_x)
        iy = self.feature_names.index(feature_y)

        X3d = self.X_train
        X2d = self.X_train_2d

        x_vals, y_vals = X2d[:, ix], X2d[:, iy]
        q = np.linspace(0, 1, bins + 1)
        x_edges = np.quantile(x_vals, q)
        y_edges = np.quantile(y_vals, q)
        dale_grid = np.zeros((bins, bins))

        for i in range(bins):
            for j in range(bins):
                x_low, x_high = x_edges[i], x_edges[i + 1]
                y_low, y_high = y_edges[j], y_edges[j + 1]
                mask = ((x_vals >= x_low) & (x_vals < x_high) & (y_vals >= y_low) & (y_vals < y_high))

                if mask.sum() == 0:
                    dale_grid[i, j] = 0
                    continue
                x_bin = tf.convert_to_tensor(X3d[mask].copy(), dtype=tf.float32)
                with tf.GradientTape() as tape:
                    tape.watch(x_bin)
                    preds = self.model(x_bin)
                grads = tape.gradient(preds, x_bin).numpy()

                # ==================================================================================================
                # [PAPER SECTION 5.1.4: First-Order DALE 2D Extension]
                # As stated in the paper: "Rather than computing mixed second-order derivatives
                # required by the original formulation, we approximate the joint effect by summing
                # univariate gradients (∂f/∂x1 + ∂f/∂x2) and accumulating them across 2D quantile bins.
                # Code Implementation:
                # g_x = univariate gradient of feature x (∂f/∂x1)
                # g_y = univariate gradient of feature y (∂f/∂x2)
                # np.mean(g_x + g_y) = The average accumulated joint effect within this specific 2D quantile bin.
                # =================================================================================================

                g_x, g_y = grads[:, ix, 0], grads[:, iy, 0]
                dale_grid[i, j] = np.mean(g_x + g_y)
        dale_accum = np.cumsum(np.cumsum(dale_grid, axis=0), axis=1)
        dale_accum -= np.mean(dale_accum)
        Xc, Yc = np.meshgrid((x_edges[:-1] + x_edges[1:]) / 2, (y_edges[:-1] + y_edges[1:]) / 2)

        fig = plt.figure(figsize=(16, 7))

        # Heatmap subplot
        ax1 = fig.add_subplot(1, 2, 1)
        hm = ax1.imshow(dale_accum, origin='lower', cmap='viridis',extent=[x_edges[0], x_edges[-1], y_edges[0], y_edges[-1]],
                        aspect='auto')
        ax1.set_title(f"DALE 2D visualization heatmap\n{feature_x} × {feature_y}", fontsize=11)
        ax1.set_xlabel(feature_x)
        ax1.set_ylabel(feature_y)
        fig.colorbar(hm, ax=ax1)

        # 3D Surface subplot
        ax2 = fig.add_subplot(1, 2, 2, projection='3d')
        ax2.plot_surface(Xc, Yc, dale_accum, cmap='viridis', edgecolor='none')
        ax2.set_title(f"DALE 3D visualization surface\n{feature_x} × {feature_y}", fontsize=11)
        ax2.set_xlabel(feature_x, labelpad=10)
        ax2.set_ylabel(feature_y, labelpad=10)
        ax2.set_zlabel("DALE Effect", labelpad=14)
        # Add axis arrows to the 3D plot
        x_span = Xc.max() - Xc.min()
        y_span = Yc.max() - Yc.min()
        z_span = dale_accum.max() - dale_accum.min()
        # Constrain vectors to occupy a clear 25% window of their baseline axis
        arrow_scale = 0.25
        # X-axis arrow (Red) - Scaled matching true X feature coordinates
        ax2.quiver(Xc.min(), Yc.min(), dale_accum.min(),x_span * arrow_scale, 0, 0,
                   length=1.0, arrow_length_ratio=0.3, color='red', linewidth=2)
        # Y-axis arrow (Green) - Scaled matching true Y feature coordinates
        ax2.quiver(Xc.min(), Yc.min(), dale_accum.min(),0, y_span * arrow_scale, 0,
                   length=1.0, arrow_length_ratio=0.3, color='green', linewidth=2)
        # Z-axis arrow (Blue) - Scaled matching target gradient units
        ax2.quiver(Xc.min(), Yc.min(), dale_accum.min(),0, 0, z_span * arrow_scale,
                   length=5.0, arrow_length_ratio=0.3, color='blue', linewidth=2)
        # Labels rendered clearly outside the arrowhead apexes
        ax2.text(Xc.min() + (x_span * arrow_scale * 1.15), Yc.min(), dale_accum.min(), 'X', color='red',
                 fontweight='bold')
        ax2.text(Xc.min(), Yc.min() + (y_span * arrow_scale * 1.15), dale_accum.min(), 'Y', color='green',
                 fontweight='bold')
        ax2.text(Xc.min(), Yc.min(), dale_accum.min() + (z_span * arrow_scale * 1.15), 'Z', color='blue',
                 fontweight='bold')
        # 3. Restructured layout parameters to prevent right-edge cropping
        plt.tight_layout(rect=(0.02, 0.02, 0.90, 0.95))
        fig.subplots_adjust(wspace=0.25)  # Balanced spacing between 2D and 3D subplots

        plt.close(fig)
        return fig

    # 3. Global Anchor Rule Formulation--------------------------------------------------------
    # Validates the precision and coverage of SHAP, PDP, and DALE-extracted rules.
    def validate_custom_anchor(self, rule_name, conditions, x_test, feature_names):
        """
        Evaluates Anchor Rules A on the untouched test set.
        Mathematically maps to:
        - Conjunctive Predicate: A = (x1 <= v1) AND (x2 >= v2) ... (Rules 1-3)
        - Coverage: cov(A) = P(z in A)  (Eq. 22)
        - Precision: prec(A) = E[1(f(x) = f(z))] (Eq. 21)
        """
        # 1. Initialize mask for the entire test set (all instances initially considered)
        mask = np.ones(len(x_test), dtype=bool)

        # 2. Apply all conditions (Conjunctive Predicate / Logical AND)
        # Translates to: A = {feature_1 <= thresh_1 and feature_2 > thresh_2 ...}
        for cond in conditions:
            if cond['feature'] not in feature_names:
                print(f"Error: Feature '{cond['feature']}' not found.")
                return 0.0, 0.0

            idx = feature_names.index(cond['feature'])
            col = x_test[:, idx]

            if cond['op'] == '<=':
                mask &= (col <= cond['threshold'])
            elif cond['op'] == '>=':
                mask &= (col >= cond['threshold'])
            elif cond['op'] == '>':
                mask &= (col > cond['threshold'])
            elif cond['op'] == '<':
                mask &= (col < cond['threshold'])
            elif cond['op'] == '==':
                mask &= (col == cond['threshold'])

        # 3. Calculate Coverage: cov(A) = P(z ~ D)(z \in A)
        # Empirically, this is the proportion of the test set that satisfies the rule.
        coverage = np.mean(mask)

        if coverage == 0:
            print(f"Rule [{rule_name}] covers 0 instances in test set.")
            return 0.0, 0.0

        # 4. Extract the subset of instances that satisfy the Anchor Rule
        x_anchor = x_test[mask]

        # Reshape for Keras (from 2D to 3D)
        if len(x_anchor.shape) == 2:
            x_anchor = x_anchor.reshape(x_anchor.shape[0], x_anchor.shape[1], 1)

        # 5. Calculate Precision: prec(A) = E_{D(z|A)}[1(f(x) = f(z))]
        # Get model predictions for the subset covered by the rule
        preds = self.model.predict(x_anchor, verbose=0).ravel()

        # Determine the majority class prediction within this rule region
        pred_classes = (preds >= 0.5).astype(int)

        # Precision: The majority class prediction within this rule region
        majority_class = 1 if np.mean(pred_classes) >= 0.5 else 0

        # Empirical precision: How often does the rule accurately predict the majority class?
        precision = np.mean(pred_classes == majority_class)

        print(f"=== {rule_name} ===")
        print(f"Coverage: {coverage * 100:.2f}% | Precision: {precision * 100:.2f}%")

        return coverage, precision
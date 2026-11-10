import base64
import webbrowser
import os
from io import BytesIO

#=======================================================================================================
def export_plots_to_html(figures, file_name="results_report.html"):

    html_code = [
        "<html>",
        "<head>",
        "<style>",
        "body { background-color: #F8F9FA; color: #212529; font-family: Arial, sans-serif; padding: 20px; }",
        "h1 { color: #000000; text-align: center; margin-bottom: 30px; }",
        "h2 { color: #0056b3; margin-top: 40px; border-bottom: 2px solid #dee2e6; padding-bottom: 10px; }",
        "img { box-shadow: 0 4px 8px rgba(0,0,0,0.1); border-radius: 8px; margin-top: 15px; }",
        "</style>",
        "</head>",
        "<body>",
        "<h1>Telecom customers churn prediction and explainability report</h1>"]
    # Plots title names list
    plots_names = ["Class distribution before and after SMOTE",
                   "Final feature columns after preprocessing",
                   "Performance metrics, confusion matrix and ROC curve of the hybrid model",
                   "Classification Report and Confusion Matrix of CNN",
                   "Classification Report and Confusion Matrix of BiLSTM",
                   "Models performance comparison table",
                   "SHAP summary plot",
                   "SHAP interaction plot: plot-1",
                   "SHAP interaction plot: plot-2",
                   "Partial dependence plot(PDP): plot-1",
                   "Partial dependence plot(PDP): plot-2",
                   "DALE: Differential Accumulated Local Effects: 2D and 3D_plot-1",
                   "DALE: Differential Accumulated Local Effects: 2D and 3D_plot-2",
                   "DALE: Differential Accumulated Local Effects: 2D and 3D_plot-3",
                   "DALE: Differential Accumulated Local Effects: 2D and 3D_plot-4",
                   "Triangulated Anchor Rules Empirical Validation"]

    for i in range(len(figures)):
        fig = figures[i]
        # Convert figure to PNG bytes
        buf = BytesIO()
        fig.savefig(buf, format="png", bbox_inches="tight", dpi=300, facecolor="white", transparent=False)
        buf.seek(0)
        img_base64 = base64.b64encode(buf.read()).decode("utf-8")

        # 1. MAJOR SECTION HEADERS STYLING
        if "SHAP summary plot" in str(plots_names[i]):
            result = ("Global Explainability" if "SHAP summary plot" in str(plots_names[i]) else "")

            html_code.append(
                f"<h2 style='text-align: center; "
                f"font-size: 26px; "
                f"font-weight: 600; "
                f"color: #FFFFFF; "
                f"background: linear-gradient(135deg, #1e3a8a, #3b82f6); "
                f"padding: 16px 20px; "
                f"border-radius: 10px; "
                f"margin-top: 70px; "
                f"margin-bottom: 40px; "
                f"box-shadow: 0 4px 12px rgba(30, 58, 138, 0.15); "
                f"border: none;'>{result}</h2>")

        # 2. Beautified individual plot titles styling --------------------------
        html_code.append(
            f"<h3 style='text-align: center; "
            f"font-size: 21px; "
            f"font-weight: 600; "
            f"color: #475569; "
            f"margin-top: 25px; "
            f"margin-bottom: 15px; "
            f"letter-spacing: 0.2px; "
            f"font-family: Arial, sans-serif;'>♦ {str(plots_names[i])} ♦</h3>")

        # 3. HIGH-RESOLUTION PLOT IMAGE DISPPLAY
        html_code.append(f'<div style="text-align: center; margin-bottom: 50px;">'
                          f'<img src="data:image/png;base64,{img_base64}" style="width:100%; max-width:1200px;">'
                          f'</div>')
        html_code.append("<hr style='border: 0; border-top: 1px solid #e2e8f0; margin: 40px 0;'>")
    html_code.append("</body></html>")

    # Save results report as html file ---------------------------------------
    with open(file_name, "w", encoding="utf-8") as f:
        f.write("\n".join(html_code))
    # Open in default browser
    abs_path = os.path.abspath(file_name)
    webbrowser.open(f"file://{abs_path}")
    print(f"Results report exported to {abs_path}")

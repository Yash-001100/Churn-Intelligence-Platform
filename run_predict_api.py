"""
Simple HTTP API to run churn predictions. Use with n8n HTTP Request node
when Execute Command is not available (e.g. n8n Cloud).

Run: python run_predict_api.py
Then call: GET or POST http://localhost:5050/run-predict
"""
import subprocess
import sys
import os

# Add project root so src is importable
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    from flask import Flask, jsonify, Response
except ImportError:
    print("Install Flask: pip install flask")
    sys.exit(1)

_report_import_error = None
try:
    from src.report_generator import generate_report_pdf
except Exception as e:
    generate_report_pdf = None
    _report_import_error = str(e)

app = Flask(__name__)
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))


@app.route("/run-predict", methods=["GET", "POST"])
def run_predict():
    """Run churn prediction script. Returns JSON with success/error."""
    try:
        result = subprocess.run(
            [sys.executable, "-m", "src.predict"],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            timeout=300,
        )
        if result.returncode == 0:
            # Add summary for email (high/medium/low counts)
            summary = None
            try:
                import pandas as pd
                pred_path = os.path.join(PROJECT_ROOT, "data", "predictions.csv")
                if os.path.exists(pred_path):
                    df = pd.read_csv(pred_path)
                    if "Risk Level" in df.columns:
                        counts = df["Risk Level"].value_counts()
                        summary = {
                            "total": len(df),
                            "high_risk": int(counts.get("High", 0)),
                            "medium_risk": int(counts.get("Medium", 0)),
                            "low_risk": int(counts.get("Low", 0)),
                        }
            except Exception:
                pass
            return jsonify({
                "success": True,
                "message": "Predictions completed. Check data/predictions.csv",
                "stdout": result.stdout.strip() or None,
                "stderr": result.stderr.strip() or None,
                "summary": summary,
            }), 200
        return jsonify({
            "success": False,
            "message": "Prediction script failed. Check stderr for details.",
            "detail": result.stderr.strip() or result.stdout.strip() or "No output",
            "stdout": result.stdout.strip() or None,
            "stderr": result.stderr.strip() or None,
            "exitCode": result.returncode,
        }), 500
    except subprocess.TimeoutExpired:
        return jsonify({
            "success": False,
            "message": "Prediction timed out (max 300s). SHAP for many high-risk customers can take longer; run predict locally or increase timeout.",
        }), 500
    except Exception as e:
        return jsonify({"success": False, "message": str(e), "detail": str(e)}), 500


@app.route("/report", methods=["GET"])
def report():
    """Generate and return PDF report (summary, charts, top high-risk table). For email attachment."""
    if generate_report_pdf is None:
        return jsonify({
            "error": "Report generator not available",
            "detail": _report_import_error or "Import failed (check reportlab, matplotlib, pandas)",
        }), 500
    pred_path = os.path.join(PROJECT_ROOT, "data", "predictions.csv")
    pred_enriched = os.path.join(PROJECT_ROOT, "data", "predictions_enriched.csv")
    path_to_use = pred_enriched if os.path.exists(pred_enriched) else pred_path
    try:
        ok, pdf_bytes = generate_report_pdf(output_path_or_buffer=None, predictions_path=path_to_use)
    except Exception as e:
        return jsonify({
            "error": "Report generation failed",
            "detail": str(e),
        }), 500
    if not ok or not pdf_bytes:
        return jsonify({
            "error": "No predictions found. Run /run-predict first.",
            "path_checked": path_to_use,
            "file_exists": os.path.exists(path_to_use),
        }), 404
    return Response(
        pdf_bytes,
        mimetype="application/pdf",
        headers={"Content-Disposition": "attachment; filename=churn-report.pdf"},
    )


@app.route("/health", methods=["GET"])
def health():
    """Health check for n8n or load balancers."""
    return jsonify({"status": "ok"}), 200


if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", 5050))
    app.run(host="0.0.0.0", port=port, debug=False)

# tests/test_train.py
import json
from pathlib import Path
from src.train import train_model


def test_training_pipeline_fast(tmp_path):
    """
    Fast integrity check for the training pipeline.
    Uses a small subset of CIFAR-10 and runs for one epoch.
    Ensures model, report, and confusion matrix artifacts are created.
    """

    # Use the pytest-provided tmp_path fixture as the output directory
    model, acc = train_model(fast_mode=True, output_dir=tmp_path)

    # Verify model object returned
    assert model is not None, "Model should not be None"

    # Define expected artifact paths
    model_path = tmp_path / "models" / "cifar-10-resnet50.pth"
    report_path = tmp_path / "assets" / "test_report.json"
    cm_path = tmp_path / "assets" / "test_confusion_matrix.png"

    # Validate artifact existence
    assert model_path.exists(), f"Expected model file at {model_path}"
    assert report_path.exists(), f"Expected JSON report at {report_path}"
    assert cm_path.exists(), f"Expected confusion matrix at {cm_path}"

    # Validate JSON report structure
    with open(report_path) as f:
        report = json.load(f)

    assert isinstance(report, dict), "Classification report should be a dict"
    assert "accuracy" in report, "Report should contain global accuracy field"

    # Accuracy should be a valid float (no NaN)
    assert isinstance(acc, float)
    assert acc >= 0.0
    assert acc <= 1.0

    # Ensure at least one class was correctly evaluated
    assert any(k in report for k in ["0", "airplane", "automobile"]), "Expected some class keys in the report"

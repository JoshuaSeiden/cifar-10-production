import json
import torch
import pytest
from src.train import train_model, set_seed


@pytest.fixture(scope="module")
def trained_model(tmp_path_factory):
    """
    Train the model once in fast_mode and return the model, accuracy, and tmp_path.
    scope="module" ensures this runs only once per test module.
    """
    tmp_path = tmp_path_factory.mktemp("train_output")
    model, acc = train_model(fast_mode=True, output_dir=tmp_path)
    return {"model": model, "acc": acc, "tmp_path": tmp_path}


def test_artifacts_created(trained_model):
    tmp_path = trained_model["tmp_path"]

    model_path = tmp_path / "models" / "cifar_10_resnet18.pth"
    report_path = tmp_path / "assets" / "test_report.json"
    cm_path = tmp_path / "assets" / "test_confusion_matrix.png"

    assert model_path.exists()
    assert report_path.exists()
    assert cm_path.exists()


def test_json_report_structure(trained_model):
    tmp_path = trained_model["tmp_path"]
    report_path = tmp_path / "assets" / "test_report.json"

    with open(report_path) as f:
        report = json.load(f)

    assert isinstance(report, dict)
    assert "accuracy" in report
    assert any(k in report for k in ["0", "airplane", "automobile"])


def test_accuracy_value(trained_model):
    acc = trained_model["acc"]
    assert isinstance(acc, float)
    assert 0.0 <= acc <= 1.0


def test_model_forward_pass(trained_model):
    model = trained_model["model"]
    model.eval()
    x = torch.randn(1, 3, 64, 64)
    with torch.no_grad():
        y = model(x)
    assert y.shape == (1, 10)

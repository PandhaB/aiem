from engine.training import checkpoint_filename, default_learning_rate, format_eta


def test_checkpoint_filename_is_zero_padded() -> None:
    assert checkpoint_filename(7, ".pth") == "model_0007.pth"
    assert checkpoint_filename(150, "stub.json") == "model_0150.stub.json"


def test_format_eta() -> None:
    assert format_eta(None) is None
    assert format_eta(0.2) is None
    assert format_eta(9) == "9s"
    assert format_eta(75) == "1m 15s"
    assert format_eta(3661) == "1h 01m"


def test_default_learning_rate() -> None:
    assert default_learning_rate("pretrained") == 0.00025
    assert default_learning_rate("random") == 0.0001

import pytest

from mlops_bits import main


def test_simple_assertion() -> None:
    """Basic sanity test to verify pytest is functioning properly."""
    assert 1 + 1 == 2


def test_main(capsys: pytest.CaptureFixture[str]) -> None:
    """Test that main() prints the expected greeting."""
    main()
    captured = capsys.readouterr()
    assert "Hello from mlops-bits!" in captured.out

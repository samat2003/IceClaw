import pytest

from iceclaw.cli import NOT_IMPLEMENTED_EXIT, main


def test_help_lists_commands(capsys):
    with pytest.raises(SystemExit) as e:
        main(["--help"])
    assert e.value.code == 0
    out = capsys.readouterr().out
    for c in ("list-regions", "investigate", "verify", "benchmark"):
        assert c in out


@pytest.mark.parametrize(
    "argv",
    [
        ["list-regions"],
        ["investigate", "--region", "r"],
        ["verify", "--run", "x"],
        ["benchmark", "--config", "c.json"],
    ],
)
def test_not_implemented(argv, capsys):
    assert main(argv) == NOT_IMPLEMENTED_EXIT
    cap = capsys.readouterr()
    assert "NOT_IMPLEMENTED" in cap.err and cap.out == ""


def test_missing_required_arg():
    with pytest.raises(SystemExit) as e:
        main(["investigate"])
    assert e.value.code != 0


def test_no_command_nonzero():
    assert main([]) != 0

import pytest

from iceclaw.cli import NOT_IMPLEMENTED_EXIT, main


def test_help_lists_commands(capsys):
    with pytest.raises(SystemExit) as e:
        main(["--help"])
    assert e.value.code == 0
    out = capsys.readouterr().out
    for c in ("list-regions", "investigate", "verify", "export", "replay", "benchmark"):
        assert c in out


def test_benchmark_is_implemented_since_p7(capsys):
    # P7 replaced the placeholder: a missing config is a specific error, not NOT_IMPLEMENTED
    assert main(["benchmark", "--config", "/nonexistent/c.json"]) == 1
    cap = capsys.readouterr()
    assert "CONFIG_NOT_FOUND" in cap.err and "NOT_IMPLEMENTED" not in cap.err
    assert NOT_IMPLEMENTED_EXIT == 2  # the exit code constant remains for future placeholders


def test_investigate_and_verify_are_implemented_not_placeholders(capsys):
    # no key / unknown run: clear, specific errors with their own exit codes (not NOT_IMPLEMENTED)
    assert main(["verify", "--run", "/nonexistent/x"]) == 2
    assert "RUN_NOT_FOUND" in capsys.readouterr().err
    assert main(["investigate", "--region", "r"]) == 2
    assert "NOT_IMPLEMENTED" not in capsys.readouterr().err


def test_missing_required_arg():
    with pytest.raises(SystemExit) as e:
        main(["investigate"])
    assert e.value.code != 0


def test_no_command_nonzero():
    assert main([]) != 0

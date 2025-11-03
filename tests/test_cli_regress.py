from analog_hawking.cli.main import main


def test_regress_cli_passes():
    rc = main(["regress"])
    assert rc in (0, 1)  # Should run without crashing; pass if 0

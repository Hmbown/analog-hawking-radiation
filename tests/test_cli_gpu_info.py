from analog_hawking.cli.main import main


def test_gpu_info_runs():
    rc = main(["gpu-info"])
    assert rc == 0

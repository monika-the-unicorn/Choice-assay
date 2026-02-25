###############################################################################
# @@@ Add a readme reference to docs explaining the structure of system test.
#
# The run_system_test script is invoked by either crontab or bcli.
# It is intended to enable nightly test runs on known samples with associated ground truth.
# This builds on top of standard pytest functionality.
#################################################################################
import pytest
from expidite_rpi.core import configuration as root_cfg

logger = root_cfg.setup_logger("expidite")


def main() -> None:
    """Run expidite-rpi in System Test mode invoking the tests defined in DeviceCfg.tests_to_run."""
    # We expect:
    # - keys.env to have the system test storage account
    # - my_device to have tests_to_run defined
    assert root_cfg.system_cfg, "system.cfg not found. Please run expidite-rpi in system test mode."
    assert root_cfg.my_device.tests_to_run, "DeviceCfg.tests_to_run not set to a list of tests to run"

    for test in root_cfg.my_device.tests_to_run:
        logger.info(f"Running system test: {test}")
        try:
            pytest.main(["-k", test, "-q", "--tb=short", "--disable-warnings"])
        except Exception:
            logger.exception("Error running test %s", test)


if __name__ == "__main__":
    main()

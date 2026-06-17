"""Installation script for the AMP_mjlab package."""

from setuptools import setup, find_packages

# Note: rsl_rl must be installed from the local copy before this package:
#   pip install -e rsl_rl/
# The local rsl_rl contains custom AMP modules (amp_ppo.py, amp_on_policy_runner.py)
# that are not in the upstream rsl-rl-lib package.

INSTALL_REQUIRES = [
    "mjlab==1.2.0",
]

setup(
    name="amp_mjlab",
    packages=find_packages(),
    version="0.0.1",
    install_requires=INSTALL_REQUIRES,
    include_package_data=True,
)

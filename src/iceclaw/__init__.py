"""IceClaw: lunar ice-prospectivity audit workbench.

Outputs are prospectivity-index audits, never detections of lunar ice.
"""

import sys

if sys.version_info < (3, 11) or sys.version_info >= (3, 13):  # pragma: no cover
    raise RuntimeError(
        "IceClaw requires Python 3.11 or 3.12; found "
        f"{sys.version_info.major}.{sys.version_info.minor}."
    )

__version__ = "0.0.1"

"""Screen the LIGHT-TUNNEL chamber datasets against the fitness gate.

Pre-registration: paper/dataset_fitness_protocol.md, light-tunnel addendum,
committed before any lag_info was computed on this data. Reuses the
wind-tunnel screen machinery unchanged; only the variable assignment and the
candidate list differ.

Settable (physical actuators the experimenter sets): red, green, blue,
pol_1, pol_2. Measured (physical consequences): current, angle_1, angle_2,
the six light sensors ir_*/vis_*, and the six wall photodiodes l_*.
Sensor-configuration parameters (osr_*, v_*, diode_*, t_*), board voltages
and metadata are excluded, as osr_*/v_*/res_* were for the wind tunnel.

    python scripts/lt_screen.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import chamber_screen as CS  # noqa: E402

CS.SETTABLE = ["red", "green", "blue", "pol_1", "pol_2"]
CS.MEASURED = ["current", "angle_1", "angle_2",
               "ir_1", "vis_1", "ir_2", "vis_2", "ir_3", "vis_3",
               "l_11", "l_12", "l_21", "l_22", "l_31", "l_32"]
CS.CANDIDATES = {
    "lt_walks_v1": "Data/causalchamber/lt_walks_v1/**/*.csv",
    "lt_test_v1": "Data/causalchamber/lt_test_v1/**/*.csv",
    "lt_interventions_standard_v1":
        "Data/causalchamber/lt_interventions_standard_v1/**/*.csv",
    "lt_malus_v1": "Data/causalchamber/lt_malus_v1/**/*.csv",
    "lt_validate_v1": "Data/causalchamber/lt_validate_v1/**/*.csv",
}
CS.OUT = Path("ExpOutput/lt_screen")

if __name__ == "__main__":
    raise SystemExit(CS.main())

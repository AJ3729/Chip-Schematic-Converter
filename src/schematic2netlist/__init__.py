"""schematic2netlist — hand-drawn circuit schematics to SPICE netlists.

Pipeline stages: preprocess -> detect -> textmask/wires -> nodes ->
snapping -> netlist -> simulate, orchestrated per-image by
:mod:`schematic2netlist.pipeline`.
"""

__version__ = "0.1.0"

from schematic2netlist.config import load_config  # noqa: F401
from schematic2netlist.pipeline import run_pipeline  # noqa: F401

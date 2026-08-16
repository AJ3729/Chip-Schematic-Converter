* Auto-generated SPICE netlist (NO TEXT OCR USED)

* SAME_NODE_SKIPPED V-AC both_on=0
* SAME_NODE_SKIPPED Diode both_on=0
* SAME_NODE_SKIPPED V-AC both_on=0
* SAME_NODE_SKIPPED Diode both_on=0
* SAME_NODE_SKIPPED V-AC both_on=0
V1 n1 0 AC 1
M1 0 0 0 0 PMOSdefault
.model PMOSdefault PMOS

* --- design-intent repair (does not change topology) ---
Rshunt_n1 n1 0 1e+09

.op
.end

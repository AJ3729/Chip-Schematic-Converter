* Auto-generated SPICE netlist (NO TEXT OCR USED)

I1 n3 0 AC 1m
* SAME_NODE_SKIPPED I-AC both_on=0
M1 0 0 0 0 PMOSdefault
* SAME_NODE_SKIPPED I-AC both_on=0
R1 0 n3 1k
* SAME_NODE_SKIPPED Resistor both_on=0
* SAME_NODE_SKIPPED Resistor both_on=0
R2 n2 0 1k
* SAME_NODE_SKIPPED Resistor both_on=0
M2 0 0 0 0 PMOSdefault
* UNSNAPPED MOSFET-P raw_nodes=[0, 1, None]
.model PMOSdefault PMOS

* --- design-intent repair (does not change topology) ---
Rshunt_n1 n1 0 1e+09
Rshunt_n2 n2 0 1e+09

.op
.end

* Auto-generated SPICE netlist (NO TEXT OCR USED)

M1 0 0 0 0 NMOSdefault
L1 0 n3 1m
L2 n2 0 1m
D1 n1 0 Zdefault
L3 0 n2 1m
* UNSNAPPED Inductor raw_nodes=[1, None]
* SAME_NODE_SKIPPED Resistor both_on=0
.model NMOSdefault NMOS
.model Zdefault D(bv=5.1)

* --- design-intent repair (does not change topology) ---
Rshunt_n1 n1 0 1e+09
Rshunt_n3 n3 0 1e+09

.op
.end

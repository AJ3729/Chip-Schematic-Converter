* Auto-generated SPICE netlist (NO TEXT OCR USED)

* SAME_NODE_SKIPPED Resistor both_on=0
R1 0 n5 1k
R2 n4 n5 1k
Q1 n2 0 0 QNPNdefault
R3 n1 n3 1k
R4 n1 n2 1k
.model QNPNdefault NPN

* --- design-intent repair (does not change topology) ---
Rshunt_n3 n3 0 1e+09
Rshunt_n4 n4 0 1e+09

.op
.end

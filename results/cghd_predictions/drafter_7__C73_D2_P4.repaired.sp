* Auto-generated SPICE netlist (NO TEXT OCR USED)

* SAME_NODE_SKIPPED Resistor both_on=0
Q1 0 n1 n3 QNPNdefault
R1 n5 n4 1k
Q2 n3 n4 n6 QNPNdefault
R2 n2 n1 1k
Q3 n6 n7 0 QNPNdefault
R3 n5 n7 1k
.model QNPNdefault NPN

* --- design-intent repair (does not change topology) ---
Rshunt_n1 n1 0 1e+09
Rshunt_n4 n4 0 1e+09

.op
.end

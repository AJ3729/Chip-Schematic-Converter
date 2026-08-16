* Auto-generated SPICE netlist (NO TEXT OCR USED)

* SAME_NODE_SKIPPED Diode both_on=n3
* SAME_NODE_SKIPPED Resistor both_on=0
Q1 n2 n4 0 QNPNdefault
R1 n1 n4 1k
Q2 n2 n5 0 QNPNdefault
.model QNPNdefault NPN

* --- design-intent repair (does not change topology) ---
Rshunt_n1 n1 0 1e+09
Rshunt_n3 n3 0 1e+09
Rshunt_n5 n5 0 1e+09

.op
.end

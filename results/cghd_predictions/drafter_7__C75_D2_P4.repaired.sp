* Auto-generated SPICE netlist (NO TEXT OCR USED)

D1 n3 n2 Ddefault
D2 0 n2 Ddefault
* SAME_NODE_SKIPPED Resistor both_on=0
* SAME_NODE_SKIPPED Resistor both_on=0
R1 0 n2 1k
Q1 n1 n2 0 QNPNdefault
.model Ddefault D
.model QNPNdefault NPN

* --- design-intent repair (does not change topology) ---
Rshunt_n1 n1 0 1e+09
Rshunt_n3 n3 0 1e+09

.op
.end

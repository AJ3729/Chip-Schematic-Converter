* Auto-generated SPICE netlist (NO TEXT OCR USED)

D1 0 n3 Ddefault
Q1 n2 n4 0 QNPNdefault
* SAME_NODE_SKIPPED Resistor both_on=n2
R1 n4 0 1k
Q2 n4 n3 n2 QNPNdefault
D2 0 n3 Ddefault
R2 n1 n3 1k
.model Ddefault D
.model QNPNdefault NPN

* --- design-intent repair (does not change topology) ---
Rshunt_n1 n1 0 1e+09

.op
.end

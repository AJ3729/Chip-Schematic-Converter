* Auto-generated SPICE netlist (NO TEXT OCR USED)

Q1 0 n1 0 QPNPdefault
R1 0 n4 1k
Q2 n3 n2 n1 QNPNdefault
.model QNPNdefault NPN
.model QPNPdefault PNP

* --- design-intent repair (does not change topology) ---
Rshunt_n1 n1 0 1e+09
Rshunt_n4 n4 0 1e+09

.op
.end

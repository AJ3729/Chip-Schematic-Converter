* Auto-generated SPICE netlist (NO TEXT OCR USED)

Q1 n1 n2 0 QNPNdefault
Q2 0 n1 0 QPNPdefault
R1 0 n3 1k
.model QNPNdefault NPN
.model QPNPdefault PNP

* --- design-intent repair (does not change topology) ---
Rshunt_n2 n2 0 1e+09
Rshunt_n3 n3 0 1e+09

.op
.end

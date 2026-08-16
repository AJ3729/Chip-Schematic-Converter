* Auto-generated SPICE netlist (NO TEXT OCR USED)

Q1 0 n1 0 QNPNdefault
Q2 n3 n2 n1 QNPNdefault
R1 0 n4 1k
.model QNPNdefault NPN

* --- design-intent repair (does not change topology) ---
Rshunt_n1 n1 0 1e+09
Rshunt_n4 n4 0 1e+09

.op
.end

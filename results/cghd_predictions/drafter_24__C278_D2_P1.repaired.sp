* Auto-generated SPICE netlist (NO TEXT OCR USED)

Q1 n2 n3 0 QNPNdefault
Q2 n2 n3 0 QNPNdefault
D1 n1 0 Ddefault
R1 n1 0 1k
.model Ddefault D
.model QNPNdefault NPN

* --- design-intent repair (does not change topology) ---
Rshunt_n3 n3 0 1e+09

.op
.end

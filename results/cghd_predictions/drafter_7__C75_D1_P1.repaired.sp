* Auto-generated SPICE netlist (NO TEXT OCR USED)

Q1 n4 0 n6 QNPNdefault
R1 n1 n2 1k
D1 n5 n2 Ddefault
R2 0 n5 1k
R3 n1 n3 1k
Q2 0 n2 n3 QNPNdefault
Q3 0 n3 n4 QNPNdefault
R4 n3 n1 1k
.model Ddefault D
.model QNPNdefault NPN

* --- design-intent repair (does not change topology) ---
Rshunt_n6 n6 0 1e+09

.op
.end

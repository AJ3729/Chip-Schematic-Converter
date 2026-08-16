* Auto-generated SPICE netlist (NO TEXT OCR USED)

Q1 n3 n4 n6 QNPNdefault
Q2 n2 0 n4 QNPNdefault
D1 n5 0 Ddefault
R1 n1 n2 1k
R2 n1 0 1k
R3 n4 n7 1k
Q3 n3 n2 n1 QNPNdefault
D2 n5 0 Ddefault
.model Ddefault D
.model QNPNdefault NPN

* --- design-intent repair (does not change topology) ---
Rshunt_n6 n6 0 1e+09
Rshunt_n7 n7 0 1e+09

.op
.end

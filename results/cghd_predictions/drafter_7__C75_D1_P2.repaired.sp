* Auto-generated SPICE netlist (NO TEXT OCR USED)

Q1 n6 n4 n8 QNPNdefault
D1 n7 n3 Ddefault
R1 n1 n2 1k
R2 n1 0 1k
R3 n4 n9 1k
Q2 0 n2 n4 QNPNdefault
D2 n3 n7 Ddefault
R4 0 n1 1k
D3 n5 n6 Ddefault
Q3 n6 n5 0 QNPNdefault
.model Ddefault D
.model QNPNdefault NPN

* --- design-intent repair (does not change topology) ---
Rshunt_n3 n3 0 1e+09
Rshunt_n8 n8 0 1e+09
Rshunt_n9 n9 0 1e+09

.op
.end

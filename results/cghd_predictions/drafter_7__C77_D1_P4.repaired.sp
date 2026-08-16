* Auto-generated SPICE netlist (NO TEXT OCR USED)

Q1 0 n2 n4 QNPNdefault
R1 0 n5 1k
R2 n1 0 1k
D1 n6 0 Ddefault
R3 n1 n2 1k
R4 n1 0 1k
R5 n1 0 1k
Q2 n4 0 n5 QNPNdefault
D2 n3 n4 Ddefault
D3 n6 0 Ddefault
.model Ddefault D
.model QNPNdefault NPN

* --- design-intent repair (does not change topology) ---
Rshunt_n3 n3 0 1e+09

.op
.end

* Auto-generated SPICE netlist (NO TEXT OCR USED)

C1 n5 n9 1u
C2 n6 n8 1u
C3 n3 n4 1u
R1 n5 n10 1k
R2 n4 0 1k
R3 n5 n6 1k
Q1 n2 n6 n7 QNPNdefault
R4 n1 n2 1k
D1 n4 n5 Ddefault
.model Ddefault D
.model QNPNdefault NPN

* --- design-intent repair (does not change topology) ---
Rshunt_n1 n1 0 1e+09
Rshunt_n3 n3 0 1e+09
Rshunt_n8 n8 0 1e+09
Rshunt_n9 n9 0 1e+09

.op
.end

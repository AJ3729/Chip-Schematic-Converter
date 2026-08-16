* Auto-generated SPICE netlist (NO TEXT OCR USED)

C1 n1 n4 1u
R1 n5 n6 1k
C2 n6 n10 1u
C3 n5 n9 1u
Q1 n3 n6 n7 QNPNdefault
R2 n5 n8 1k
R3 n4 0 1k
R4 n2 n3 1k
D1 n4 n5 Ddefault
.model Ddefault D
.model QNPNdefault NPN

* --- design-intent repair (does not change topology) ---
Rshunt_n1 n1 0 1e+09
Rshunt_n10 n10 0 1e+09
Rshunt_n11 n11 0 1e+09
Rshunt_n2 n2 0 1e+09
Rshunt_n8 n8 0 1e+09
Rshunt_n9 n9 0 1e+09

.op
.end

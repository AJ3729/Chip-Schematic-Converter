* Auto-generated SPICE netlist (NO TEXT OCR USED)

C1 0 n8 1u
R1 n1 0 1k
R2 n6 n8 1k
R3 n8 n5 1k
R4 n1 n5 1k
R5 n1 n4 1k
R6 n2 n3 1k
C2 0 n5 1u
C3 n4 n7 1u
D1 n7 0 Ddefault
D2 n5 n3 Ddefault
D3 n5 0 Ddefault
.model Ddefault D

* --- design-intent repair (does not change topology) ---
Rshunt_n2 n2 0 1e+09
Rshunt_n6 n6 0 1e+09

.op
.end

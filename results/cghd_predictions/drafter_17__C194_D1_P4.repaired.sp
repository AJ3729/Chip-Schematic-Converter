* Auto-generated SPICE netlist (NO TEXT OCR USED)

E1 n11 0 n10 n5 100k
C1 n2 n4 1u
E2 n3 0 n9 n5 100k
C2 n7 n8 1u
R1 n7 n3 1k
R2 n5 n9 1k
R3 n9 n3 1k
R4 n9 n5 1k
R5 n10 n5 1k
R6 n5 0 1k
R7 n8 n3 1k
R8 n10 0 1k
C3 n5 0 1u
C4 n8 n9 1u
R9 n11 0 1k
R10 n4 n3 1k
D1 n5 n6 Ddefault
C5 n6 n5 1u
C6 n4 n7 1u
R11 n2 n3 1k
C7 n3 n12 1u
C8 n11 0 1u
V1 n5 0 DC 5
V2 n12 0 DC 5
.model Ddefault D

* --- design-intent repair (does not change topology) ---
Rshunt_n1 n1 0 1e+09
Rshunt_n12 n12 0 1e+09

.op
.end

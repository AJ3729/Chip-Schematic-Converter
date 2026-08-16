* Auto-generated SPICE netlist (NO TEXT OCR USED)

E1 n1 0 n12 n1 100k
E2 n1 0 n1 n2 100k
D1 n4 n14 Ddefault
R1 n3 n2 1k
R2 n11 n4 1k
R3 n5 n3 1k
R4 n5 n1 1k
C1 n2 n1 1u
L1 n6 n4 1m
R5 n10 n13 1k
E3 n3 0 n1 n5 100k
I1 n11 n14 AC 1m
E4 n6 0 n1 n8 100k
L2 n12 n15 1m
R6 n7 n10 1k
R7 n9 n12 1k
R8 n13 0 1k
.model Ddefault D

* --- design-intent repair (does not change topology) ---
Rshunt_n1 n1 0 1e+09
Rshunt_n7 n7 0 1e+09

.op
.end

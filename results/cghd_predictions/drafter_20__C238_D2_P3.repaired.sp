* Auto-generated SPICE netlist (NO TEXT OCR USED)

E1 n4 0 n4 n14 100k
E2 n8 0 n10 n3 100k
E3 n3 0 n2 n4 100k
D1 n7 n16 Ddefault
I1 n15 n16 AC 1m
R1 n6 n3 1k
E4 n5 0 n4 n6 100k
R2 n15 n7 1k
R3 n5 n2 1k
C1 n2 n3 1u
R4 n13 0 1k
L1 n8 n7 1m
R5 n14 n17 1k
R6 n6 n5 1k
R7 n12 n9 1k
R8 n11 n14 1k
R9 n13 n10 1k
V1 n7 0 DC 5
.model Ddefault D

* --- design-intent repair (does not change topology) ---
Rshunt_n1 n1 0 1e+09
Rshunt_n11 n11 0 1e+09
Rshunt_n12 n12 0 1e+09

.op
.end

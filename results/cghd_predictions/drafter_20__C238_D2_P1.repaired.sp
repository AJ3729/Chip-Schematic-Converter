* Auto-generated SPICE netlist (NO TEXT OCR USED)

E1 n3 0 n13 n3 100k
E2 n1 0 n2 n3 100k
E3 n4 0 n5 n3 100k
E4 n7 0 n8 n1 100k
R1 n5 n1 1k
R2 n12 n6 1k
R3 n4 n2 1k
R4 n5 n4 1k
L1 n7 n6 1m
R5 n13 n15 1k
D1 n6 n14 Ddefault
C1 n2 n1 1u
R6 n3 n13 1k
R7 n11 0 1k
I1 n12 n14 AC 1m
R8 n9 n10 1k
R9 n11 n8 1k
V1 n6 0 DC 5
V2 n6 0 DC 5
.model Ddefault D

* --- design-intent repair (does not change topology) ---
Rshunt_n10 n10 0 1e+09

.op
.end

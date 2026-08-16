* Auto-generated SPICE netlist (NO TEXT OCR USED)

E1 n1 0 n5 n2 100k
E2 n3 0 n6 n5 100k
R1 n3 n2 1k
D1 n13 n4 Ddefault
R2 n6 n1 1k
C1 n2 n1 1u
L1 n7 n4 1m
R3 n11 n4 1k
R4 n6 n3 1k
R5 n8 0 1k
E3 n5 0 n12 n5 100k
E4 n7 0 n9 n1 100k
I1 n11 n13 AC 1m
L2 n10 n12 1m
.model Ddefault D

.op
.end

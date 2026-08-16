* Auto-generated SPICE netlist (NO TEXT OCR USED)

E1 n4 0 n3 n5 100k
R1 0 n6 1k
R2 n3 0 1k
R3 n5 n6 1k
R4 0 n3 1k
R5 0 n1 1k
E2 n6 0 n6 0 100k
L1 n2 0 1m
D1 n4 n1 Ddefault
D2 n4 n1 Ddefault
C1 n2 0 1u
.model Ddefault D

.op
.end

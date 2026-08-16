* Auto-generated SPICE netlist (NO TEXT OCR USED)

I1 n3 n6 AC 1m
E1 n4 0 n1 n5 100k
R1 n10 n7 1k
R2 n4 n2 1k
C1 n2 n1 1u
R3 n5 n4 1k
R4 n5 n1 1k
R5 n1 n14 1k
L1 n1 n6 1m
I2 n10 n12 AC 1m
R6 n11 0 1k
R7 n8 n9 1k
E2 n1 0 n1 n2 100k
R8 n11 n1 1k
I3 n10 n12 DC 1m
R9 n13 n1 1k
D1 n7 n12 Ddefault
D2 n7 n12 Ddefault
.model Ddefault D

.op
.end

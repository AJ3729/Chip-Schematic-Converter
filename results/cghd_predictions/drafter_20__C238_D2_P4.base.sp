* Auto-generated SPICE netlist (NO TEXT OCR USED)

I1 n14 n17 AC 1m
E1 n4 0 n16 n4 100k
E2 n3 0 n4 n6 100k
E3 n8 0 n9 n1 100k
E4 n1 0 n4 n2 100k
D1 n11 n17 Ddefault
R1 n3 n2 1k
R2 n14 n11 1k
R3 n6 n1 1k
R4 n6 n3 1k
C1 n2 n1 1u
L1 n8 n7 1m
R5 n16 0 1k
R6 n15 n18 1k
R7 n4 n16 1k
I2 n7 n5 AC 1m
R8 n12 n10 1k
R9 n13 n9 1k
.model Ddefault D

.op
.end

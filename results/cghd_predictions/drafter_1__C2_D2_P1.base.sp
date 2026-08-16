* Auto-generated SPICE netlist (NO TEXT OCR USED)

R1 n1 n7 1k
I1 n2 n1 DC 1m
V1 n8 0 DC 5
C1 n4 0 1u
C2 n7 n8 1u
R2 n5 0 1k
D1 n7 n3 Ddefault
R3 0 n6 1k
.model Ddefault D

.op
.end

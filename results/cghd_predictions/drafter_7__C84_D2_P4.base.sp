* Auto-generated SPICE netlist (NO TEXT OCR USED)

R1 n1 0 1k
R2 n2 n3 1k
R3 n1 n2 1k
R4 0 n3 1k
I1 0 n3 DC 1m
R5 n1 0 1k
R6 n3 n4 1k
D1 n4 0 Ddefault
C1 0 n5 1u
V1 n2 0 DC 5
C2 n4 n5 1u
I2 n1 0 AC 1m
.model Ddefault D

.op
.end

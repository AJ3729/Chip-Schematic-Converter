* Auto-generated SPICE netlist (NO TEXT OCR USED)

R1 n3 n4 1k
R2 n4 n1 1k
I1 n4 n1 AC 1m
R3 n2 n1 1k
R4 n3 n2 1k
R5 n6 n8 1k
R6 n1 0 1k
I2 n7 n5 AC 1m
I3 n5 n7 DC 1m
D1 n8 n5 Ddefault
V1 n2 0 DC 5
.model Ddefault D

.op
.end

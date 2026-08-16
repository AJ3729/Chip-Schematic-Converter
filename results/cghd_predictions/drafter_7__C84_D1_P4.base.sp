* Auto-generated SPICE netlist (NO TEXT OCR USED)

R1 0 n1 1k
R2 0 n3 1k
R3 n3 n2 1k
R4 n1 n2 1k
R5 0 n3 1k
R6 n2 n4 1k
I1 n3 0 DC 1m
V1 n1 0 DC 5
I2 n3 n2 DC 1m
I3 n3 n2 AC 1m
I4 0 n3 AC 1m
D1 n4 n3 Ddefault
.model Ddefault D

.op
.end

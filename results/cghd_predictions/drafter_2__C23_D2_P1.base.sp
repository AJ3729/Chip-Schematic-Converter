* Auto-generated SPICE netlist (NO TEXT OCR USED)

R1 n1 n8 1k
C1 n8 0 1u
V1 n12 n2 DC 5
C2 n9 0 1u
I1 n8 n10 DC 1m
C3 n10 n5 1u
R2 n1 n6 1k
R3 n1 n5 1k
R4 n1 n7 1k
D1 0 n11 Ddefault
R5 n4 0 1k
D2 n9 0 Zdefault
R6 n3 0 1k
D3 0 n10 Ddefault
I2 n7 0 DC 1m
D4 n10 0 Zdefault
.model Ddefault D
.model Zdefault D(bv=5.1)

.op
.end

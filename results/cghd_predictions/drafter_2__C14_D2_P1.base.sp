* Auto-generated SPICE netlist (NO TEXT OCR USED)

I1 0 n5 AC 1m
V1 0 n2 DC 5
R1 n1 n5 1k
I2 n5 n6 DC 1m
D1 n1 n3 Zdefault
R2 n3 n5 1k
R3 n1 n5 1k
R4 n4 n6 1k
C1 n5 n6 1u
D2 n3 n1 Ddefault
D3 n4 n1 Ddefault
.model Ddefault D
.model Zdefault D(bv=5.1)

.op
.end

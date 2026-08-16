* Auto-generated SPICE netlist (NO TEXT OCR USED)

I1 n2 n4 AC 1m
R1 n2 0 1k
D1 0 n3 Zdefault
R2 n3 0 1k
R3 n3 n4 1k
D2 n1 0 Ddefault
R4 n3 n4 1k
R5 n1 0 1k
.model Ddefault D
.model Zdefault D(bv=5.1)

.op
.end

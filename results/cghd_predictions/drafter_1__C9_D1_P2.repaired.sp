* Auto-generated SPICE netlist (NO TEXT OCR USED)

D1 n3 0 Zdefault
R1 n1 n2 1k
C1 n2 0 1u
R2 n2 n3 1k
V1 n1 0 DC 5
V2 n1 0 DC 5
I1 n1 0 AC 1m
.model Zdefault D(bv=5.1)

.op
.end

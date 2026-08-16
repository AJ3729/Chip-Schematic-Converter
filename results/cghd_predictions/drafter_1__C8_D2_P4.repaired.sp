* Auto-generated SPICE netlist (NO TEXT OCR USED)

V1 n1 0 DC 5
R1 0 n1 1k
R2 n2 n1 1k
E1 0 0 n3 n2 100k
C1 n2 0 1u
D1 n1 n3 Zdefault
R3 0 n1 1k
D2 n3 0 Zdefault
.model Zdefault D(bv=5.1)

.op
.end

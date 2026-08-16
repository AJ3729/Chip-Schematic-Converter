* Auto-generated SPICE netlist (NO TEXT OCR USED)

D1 0 n2 Zdefault
C1 n1 n3 1u
V1 0 n3 DC 5
E1 n3 0 n2 n1 100k
R1 n1 0 1k
R2 n3 0 1k
C2 n2 0 1u
V2 0 n3 DC 5
R3 n3 0 1k
.model Zdefault D(bv=5.1)

.op
.end

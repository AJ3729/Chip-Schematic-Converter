* Auto-generated SPICE netlist (NO TEXT OCR USED)

V1 n2 0 DC 5
R1 n2 n1 1k
C1 n1 0 1u
D1 0 n3 Ddefault
R2 n1 n3 1k
D2 n3 0 Zdefault
.model Ddefault D
.model Zdefault D(bv=5.1)

.op
.end

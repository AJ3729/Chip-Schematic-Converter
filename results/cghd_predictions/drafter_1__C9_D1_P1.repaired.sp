* Auto-generated SPICE netlist (NO TEXT OCR USED)

V1 n1 0 DC 5
C1 n2 0 1u
R1 n1 n2 1k
R2 n2 n3 1k
D1 n3 0 Zdefault
D2 0 n3 Ddefault
.model Ddefault D
.model Zdefault D(bv=5.1)

.op
.end

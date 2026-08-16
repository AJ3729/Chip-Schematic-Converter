* Auto-generated SPICE netlist (NO TEXT OCR USED)

V1 0 n1 DC 5
C1 n2 0 1u
R1 n1 n2 1k
D1 n3 0 Zdefault
R2 n2 n3 1k
D2 n3 0 Ddefault
.model Ddefault D
.model Zdefault D(bv=5.1)

.op
.end

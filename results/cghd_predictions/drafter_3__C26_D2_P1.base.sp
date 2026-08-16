* Auto-generated SPICE netlist (NO TEXT OCR USED)

R1 0 n1 1k
C1 0 n1 1u
R2 n2 n3 1k
D1 0 n2 Ddefault
D2 n2 n1 Zdefault
.model Ddefault D
.model Zdefault D(bv=5.1)

.op
.end

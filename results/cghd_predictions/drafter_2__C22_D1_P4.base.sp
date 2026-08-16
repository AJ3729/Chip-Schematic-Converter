* Auto-generated SPICE netlist (NO TEXT OCR USED)

V1 0 n1 DC 5
R1 n1 0 1k
R2 n4 0 1k
R3 n3 0 1k
D1 n2 0 Zdefault
R4 n1 n2 1k
C1 0 n3 1u
C2 n4 n3 1u
C3 0 n4 1u
D2 0 n2 Ddefault
.model Ddefault D
.model Zdefault D(bv=5.1)

.op
.end

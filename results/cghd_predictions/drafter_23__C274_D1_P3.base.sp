* Auto-generated SPICE netlist (NO TEXT OCR USED)

C1 n1 n2 1u
L1 n6 n2 1m
D1 n5 n2 Ddefault
E1 n2 0 n4 n3 100k
D2 n4 n1 Zdefault
C2 n6 n2 1u
.model Ddefault D
.model Zdefault D(bv=5.1)

.op
.end

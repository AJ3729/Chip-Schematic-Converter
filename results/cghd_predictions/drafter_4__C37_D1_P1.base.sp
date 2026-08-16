* Auto-generated SPICE netlist (NO TEXT OCR USED)

R1 n12 0 1k
C1 n4 n12 1u
R2 n6 n12 1k
R3 n1 0 1k
R4 n2 n5 1k
R5 n1 n3 1k
R6 n1 n4 1k
D1 n5 n10 Ddefault
D2 0 n8 Zdefault
D3 n7 n11 Ddefault
D4 n4 n9 Zdefault
C2 n4 0 1u
M1 n4 n9 n3 n3 PMOSdefault
M2 n3 n6 n7 n7 PMOSdefault
C3 0 n5 1u
.model Ddefault D
.model PMOSdefault PMOS
.model Zdefault D(bv=5.1)

.op
.end

* Auto-generated SPICE netlist (NO TEXT OCR USED)

V1 n3 n1 DC 5
R1 n1 n3 1k
R2 n4 n3 1k
R3 n1 n2 1k
D1 n2 n3 Zdefault
R4 0 n3 1k
M1 n4 n2 n3 n3 PMOSdefault
C1 n3 n4 1u
C2 n3 n4 1u
.model PMOSdefault PMOS
.model Zdefault D(bv=5.1)

.op
.end

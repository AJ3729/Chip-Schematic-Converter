* Auto-generated SPICE netlist (NO TEXT OCR USED)

C1 n5 n9 1u
R1 n9 0 1k
R2 n7 n9 1k
R3 n3 n5 1k
R4 n3 n6 1k
R5 n3 0 1k
R6 n2 n4 1k
* UNSNAPPED BJT-NPN raw_nodes=[3, 0, None]
D1 n9 0 Ddefault
C2 n6 n5 1u
C3 n5 0 1u
M1 n8 n6 n7 n7 PMOSdefault
C4 0 n4 1u
.model Ddefault D
.model PMOSdefault PMOS

.op
.end

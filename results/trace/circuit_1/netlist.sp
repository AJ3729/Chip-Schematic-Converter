* Auto-generated SPICE netlist (NO TEXT OCR USED)

M1 n5 n6 n7 n7 PMOSdefault
M2 n5 n4 n9 n9 NMOSdefault
I1 n6 0 AC 1m
M3 n2 n5 n4 n4 PMOSdefault
M4 n2 n3 n5 n5 NMOSdefault
V1 n1 n2 DC 5
I2 n8 0 AC 1m
V2 n3 0 AC 1
V3 n4 0 AC 1
R1 n9 n7 1k
D1 n2 n4 Zdefault
M5 n9 n3 0 0 NMOSdefault
M6 n7 0 n8 n8 NMOSdefault
.model NMOSdefault NMOS
.model PMOSdefault PMOS
.model Zdefault D(bv=5.1)

.op
.end

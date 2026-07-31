* Auto-generated SPICE netlist (NO TEXT OCR USED)

M1 n3 n1 n4 n4 PMOSdefault
V1 n1 0 DC 5
V2 n2 n9 AC 1
D1 n1 n3 Zdefault
V3 n3 n9 AC 1
M2 n4 n3 n7 n7 NMOSdefault
R1 n7 n6 1k
M3 n7 n2 n9 n9 NMOSdefault
M4 n6 n9 n8 n8 NMOSdefault
I1 n5 n9 DC 1m
M5 n1 n2 n4 n4 NMOSdefault
M6 n4 n6 n5 n5 NMOSdefault
I2 n8 n9 AC 1m
I3 n5 n9 AC 1m
.model NMOSdefault NMOS
.model PMOSdefault PMOS
.model Zdefault D(bv=5.1)

.op
.end

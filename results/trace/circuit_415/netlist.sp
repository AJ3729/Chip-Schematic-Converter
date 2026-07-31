* Auto-generated SPICE netlist (NO TEXT OCR USED)

I1 n8 n10 AC 1m
I2 n6 n10 AC 1m
M1 n6 n4 n5 n5 PMOSdefault
M2 n1 n4 n2 n2 PMOSdefault
M3 n5 n8 n10 n10 PMOSdefault
V1 n1 0 DC 5
V2 n3 n9 AC 1
V3 n2 n9 AC 1
D1 n1 n2 Zdefault
M4 n1 n3 n4 n4 NMOSdefault
M5 n7 n3 n9 n9 NMOSdefault
R1 n7 n5 1k
M6 n4 n2 n7 n7 NMOSdefault
M7 n1 n4 n3 n3 PMOSdefault
M8 n4 n7 n2 n2 PMOSdefault
.model NMOSdefault NMOS
.model PMOSdefault PMOS
.model Zdefault D(bv=5.1)

.op
.end

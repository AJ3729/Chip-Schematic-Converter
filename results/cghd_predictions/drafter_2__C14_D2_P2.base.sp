* Auto-generated SPICE netlist (NO TEXT OCR USED)

D1 0 n1 Zdefault
D2 0 n2 Zdefault
V1 n10 0 DC 5
I1 n10 n9 DC 1m
R1 n3 n4 1k
I2 n9 n11 DC 1m
R2 n6 n7 1k
R3 0 n9 1k
D3 n2 0 Ddefault
R4 n5 n3 1k
D4 n1 0 Ddefault
M1 n8 n9 n8 n8 PMOSdefault
* SAME_NODE_SKIPPED Capacitor both_on=n10
.model Ddefault D
.model PMOSdefault PMOS
.model Zdefault D(bv=5.1)

.op
.end

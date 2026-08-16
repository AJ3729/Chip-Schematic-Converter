* Auto-generated SPICE netlist (NO TEXT OCR USED)

R1 n3 n1 1k
Q1 n6 0 n7 QNPNdefault
Q2 0 n4 n3 QNPNdefault
R2 n1 n2 1k
R3 n1 0 1k
* SAME_NODE_SKIPPED Resistor both_on=0
R4 n1 n3 1k
* SAME_NODE_SKIPPED V-DC (one port) both_on=0
Q3 n5 n2 0 QNPNdefault
* SAME_NODE_SKIPPED V-DC (one port) both_on=0
M1 n3 0 n5 n5 PMOSdefault
D1 n6 n4 Ddefault
.model Ddefault D
.model PMOSdefault PMOS
.model QNPNdefault NPN

.op
.end

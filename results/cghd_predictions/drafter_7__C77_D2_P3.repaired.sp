* Auto-generated SPICE netlist (NO TEXT OCR USED)

Q1 0 n3 n2 QNPNdefault
Q2 n4 0 0 QNPNdefault
R1 n2 n1 1k
* SAME_NODE_SKIPPED Resistor both_on=0
R2 n1 0 1k
R3 n1 n2 1k
R4 n1 0 1k
M1 0 0 n2 n2 PMOSdefault
D1 n4 n3 Ddefault
.model Ddefault D
.model PMOSdefault PMOS
.model QNPNdefault NPN

.op
.end

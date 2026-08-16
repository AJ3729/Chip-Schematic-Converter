* Auto-generated SPICE netlist (NO TEXT OCR USED)

D1 n6 0 Ddefault
D2 n6 0 Ddefault
Q1 n7 n8 n6 QNPNdefault
R1 0 n7 1k
R2 n3 n4 1k
R3 n5 0 1k
R4 0 n8 1k
R5 n3 0 1k
Q2 n4 0 0 QNPNdefault
* SAME_NODE_SKIPPED Resistor both_on=0
R6 n3 0 1k
R7 n2 n1 1k
* SAME_NODE_SKIPPED V-DC (one port) both_on=0
Q3 0 n4 0 QNPNdefault
C1 n2 n1 1u
.model Ddefault D
.model QNPNdefault NPN

.op
.end

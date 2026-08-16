* Auto-generated SPICE netlist (NO TEXT OCR USED)

Q1 n3 n2 0 QNPNdefault
R1 n1 n2 1k
R2 n1 0 1k
Q2 0 n5 n4 QNPNdefault
* SAME_NODE_SKIPPED Resistor both_on=0
R3 n1 0 1k
R4 n1 0 1k
D1 n3 n4 Ddefault
.model Ddefault D
.model QNPNdefault NPN

.op
.end

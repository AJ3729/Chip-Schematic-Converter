* Auto-generated SPICE netlist (NO TEXT OCR USED)

Q1 n3 n4 n5 QNPNdefault
* UNSNAPPED BJT-NPN raw_nodes=[4, 3, None]
D1 n5 0 Ddefault
R1 n1 0 1k
R2 n1 n2 1k
R3 n4 n5 1k
R4 n1 n2 1k
D2 n5 0 Ddefault
Q2 n2 0 n4 QNPNdefault
* SAME_NODE_SKIPPED Resistor both_on=0
.model Ddefault D
.model QNPNdefault NPN

.op
.end

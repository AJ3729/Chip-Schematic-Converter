* Auto-generated SPICE netlist (NO TEXT OCR USED)

D1 0 n2 Ddefault
Q1 n3 n4 0 QNPNdefault
Q2 n2 n2 n2 QNPNdefault
D2 0 n2 Ddefault
R1 n2 n4 1k
R2 n1 n2 1k
* SAME_NODE_SKIPPED Zener Diode both_on=n2
R3 n1 n2 1k
R4 n1 n2 1k
* SAME_NODE_SKIPPED Resistor both_on=n2
R5 n2 n3 1k
* SAME_NODE_SKIPPED Resistor both_on=n2
* SAME_NODE_SKIPPED Zener Diode both_on=n2
* SAME_NODE_SKIPPED Zener Diode both_on=n2
.model Ddefault D
.model QNPNdefault NPN

.op
.end

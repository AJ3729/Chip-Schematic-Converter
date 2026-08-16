* Auto-generated SPICE netlist (NO TEXT OCR USED)

C1 n1 0 1u
* SAME_NODE_SKIPPED Resistor both_on=n4
* SAME_NODE_SKIPPED Resistor both_on=n4
E1 n1 0 n3 n1 100k
Q1 n4 n4 n4 QNPNdefault
E2 n2 0 n3 n4 100k
.model QNPNdefault NPN

.op
.end

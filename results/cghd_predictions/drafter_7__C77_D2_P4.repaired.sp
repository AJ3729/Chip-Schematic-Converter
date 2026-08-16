* Auto-generated SPICE netlist (NO TEXT OCR USED)

R1 n1 n3 1k
* SAME_NODE_SKIPPED Resistor both_on=0
R2 n2 0 1k
R3 n2 0 1k
Q1 n1 n3 0 QNPNdefault
M1 0 0 0 0 PMOSdefault
* SAME_NODE_SKIPPED Diode both_on=0
.model PMOSdefault PMOS
.model QNPNdefault NPN

.op
.end

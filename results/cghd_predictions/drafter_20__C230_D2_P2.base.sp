* Auto-generated SPICE netlist (NO TEXT OCR USED)

Q1 n6 0 n10 QNPNdefault
Q2 n1 n2 n4 QNPNdefault
Q3 0 n6 n7 QPNPdefault
R1 n7 n9 1k
R2 n6 n5 1k
* SAME_NODE_SKIPPED Resistor both_on=n2
C1 n4 n3 1u
R3 n10 0 1k
* SAME_NODE_SKIPPED Resistor both_on=0
R4 n4 0 1k
R5 n1 n2 1k
R6 n10 0 1k
R7 n11 0 1k
C2 n7 n11 1u
R8 n4 n6 1k
* SAME_NODE_SKIPPED Resistor both_on=0
C3 n12 0 1u
C4 n4 n3 1u
V1 0 n12 AC 1
* SAME_NODE_SKIPPED Capacitor both_on=n1
C5 n9 n8 1u
I1 0 n8 DC 1m
M1 n4 n5 n7 n7 PMOSdefault
.model PMOSdefault PMOS
.model QNPNdefault NPN
.model QPNPdefault PNP

.op
.end

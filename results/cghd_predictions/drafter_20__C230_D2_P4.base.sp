* Auto-generated SPICE netlist (NO TEXT OCR USED)

Q1 n7 0 n11 QNPNdefault
C1 n13 0 1u
Q2 n9 n7 0 QPNPdefault
Q3 n1 n3 n5 QNPNdefault
R1 n9 n10 1k
R2 n5 0 1k
* SAME_NODE_SKIPPED Resistor both_on=n3
R3 n5 n7 1k
R4 n7 n6 1k
R5 n11 0 1k
R6 n1 n3 1k
V1 0 n13 AC 1
* SAME_NODE_SKIPPED Resistor both_on=0
R7 n12 0 1k
* UNSNAPPED BJT-PNP raw_nodes=[None, None, None]
C2 n9 n12 1u
R8 n11 0 1k
R9 n14 0 1k
C3 n5 n4 1u
C4 n1 n2 1u
C5 n5 n4 1u
M1 n5 n6 n9 n9 PMOSdefault
I1 0 n8 DC 1m
C6 n1 n2 1u
C7 n10 n8 1u
.model PMOSdefault PMOS
.model QNPNdefault NPN
.model QPNPdefault PNP

.op
.end

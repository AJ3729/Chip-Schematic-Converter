* Auto-generated SPICE netlist (NO TEXT OCR USED)

Q1 n8 0 n13 QNPNdefault
C1 n1 n2 1u
R1 n8 n7 1k
C2 n10 n12 1u
Q2 n1 n3 n5 QNPNdefault
C3 n5 n4 1u
Q3 n10 n8 0 QPNPdefault
R2 n3 n6 1k
R3 n13 n15 1k
R4 n5 0 1k
C4 n3 n6 1u
R5 n10 n11 1k
R6 n14 0 1k
R7 n13 0 1k
* SAME_NODE_SKIPPED Resistor both_on=0
R8 n5 n8 1k
R9 n1 n3 1k
C5 n1 n2 1u
R10 n12 0 1k
C6 n15 0 1u
V1 n9 0 AC 1
* SAME_NODE_SKIPPED I-DC both_on=0
C7 n11 n9 1u
C8 n5 n4 1u
Q4 n7 n5 n10 QPNPdefault
V2 n2 0 DC 5
.model QNPNdefault NPN
.model QPNPdefault PNP

.op
.end

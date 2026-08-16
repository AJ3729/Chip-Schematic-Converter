* Auto-generated SPICE netlist (NO TEXT OCR USED)

Q1 n2 n5 n9 QNPNdefault
Q2 n5 n5 n7 QNPNdefault
Q3 n1 n2 n2 QPNPdefault
Q4 0 0 0 QPNPdefault
C1 n3 n2 1u
* SAME_NODE_SKIPPED Resistor both_on=0
R1 n2 n7 1k
R2 0 n1 1k
Q5 0 n6 n4 QNPNdefault
Q6 0 n3 n7 QNPNdefault
R3 n6 n8 1k
R4 0 n6 1k
Q7 n4 n5 n7 QNPNdefault
Q8 0 n2 n3 QPNPdefault
R5 n9 n7 1k
R6 n3 n7 1k
L1 0 n5 1m
Q9 0 0 n4 QNPNdefault
C2 n2 0 1u
Q10 n2 0 0 QPNPdefault
R7 0 n5 1k
Q11 n8 n7 n7 QPNPdefault
R8 0 n5 1k
.model QNPNdefault NPN
.model QPNPdefault PNP

.op
.end

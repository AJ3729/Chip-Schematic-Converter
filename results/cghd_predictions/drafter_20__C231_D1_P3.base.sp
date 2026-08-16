* Auto-generated SPICE netlist (NO TEXT OCR USED)

Q1 0 n4 n8 QNPNdefault
Q2 n5 n6 n8 QNPNdefault
Q3 0 n2 n4 QPNPdefault
Q4 n6 n6 n8 QNPNdefault
R1 n4 n8 1k
Q5 n2 n3 n9 QNPNdefault
Q6 0 n7 n5 QNPNdefault
C1 n4 n2 1u
Q7 n3 n6 n10 QNPNdefault
R2 0 n1 1k
R3 n9 n8 1k
Q8 0 0 n5 QNPNdefault
R4 n7 n8 1k
R5 n10 n8 1k
C2 n3 0 1u
R6 0 n7 1k
R7 0 n6 1k
* SAME_NODE_SKIPPED Resistor both_on=0
Q9 0 0 0 QPNPdefault
Q10 n3 0 0 QPNPdefault
Q11 n1 n2 n1 QPNPdefault
.model QNPNdefault NPN
.model QPNPdefault PNP

.op
.end

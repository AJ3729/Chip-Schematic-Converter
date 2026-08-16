* Auto-generated SPICE netlist (NO TEXT OCR USED)

Q1 0 0 n5 QNPNdefault
Q2 n6 n6 n8 QNPNdefault
Q3 n5 n6 n8 QNPNdefault
Q4 0 n4 n8 QNPNdefault
Q5 0 n7 n5 QNPNdefault
Q6 n2 n3 n10 QNPNdefault
C1 n4 n2 1u
Q7 n3 n6 n11 QNPNdefault
R1 0 n2 1k
R2 n4 n8 1k
* SAME_NODE_SKIPPED Resistor both_on=0
R3 0 n6 1k
Q8 n3 0 0 QPNPdefault
R4 n7 n9 1k
Q9 0 n2 n4 QPNPdefault
* UNSNAPPED Inductor raw_nodes=[0, None]
R5 n10 n8 1k
C2 n3 0 1u
R6 0 n7 1k
Q10 0 0 0 QPNPdefault
Q11 n9 n8 n9 QPNPdefault
R7 n11 n8 1k
.model QNPNdefault NPN
.model QPNPdefault PNP

.op
.end

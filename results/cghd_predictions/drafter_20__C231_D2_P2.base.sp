* Auto-generated SPICE netlist (NO TEXT OCR USED)

Q1 0 n6 0 QNPNdefault
C1 n3 n2 1u
Q2 0 n5 n4 QNPNdefault
Q3 0 0 0 QPNPdefault
R1 0 n5 1k
R2 0 n1 1k
Q4 n1 n2 n2 QPNPdefault
* SAME_NODE_SKIPPED Resistor both_on=0
Q5 0 0 n4 QNPNdefault
R3 n5 n6 1k
R4 n7 n6 1k
Q6 0 0 n8 QNPNdefault
Q7 n3 n2 0 QPNPdefault
R5 n3 n6 1k
Q8 n4 0 n6 QNPNdefault
Q9 0 n3 n6 QNPNdefault
R6 n8 n6 1k
* SAME_NODE_SKIPPED Capacitor both_on=0
Q10 n2 0 n7 QNPNdefault
.model QNPNdefault NPN
.model QPNPdefault PNP

.op
.end

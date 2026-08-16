* Auto-generated SPICE netlist (NO TEXT OCR USED)

Q1 n4 n4 n6 QNPNdefault
C1 n3 n2 1u
Q2 0 n5 0 QNPNdefault
Q3 0 0 0 QPNPdefault
R1 0 n1 1k
R2 0 n5 1k
R3 n7 n6 1k
* SAME_NODE_SKIPPED Resistor both_on=0
Q4 n1 n2 n2 QPNPdefault
R4 n5 n6 1k
R5 0 n4 1k
Q5 0 n4 n6 QNPNdefault
* SAME_NODE_SKIPPED Capacitor both_on=0
Q6 0 n2 n3 QPNPdefault
R6 n3 n6 1k
Q7 0 n3 n6 QNPNdefault
Q8 0 n4 n6 QNPNdefault
Q9 n2 0 n7 QNPNdefault
.model QNPNdefault NPN
.model QPNPdefault PNP

.op
.end

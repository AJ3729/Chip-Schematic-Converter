* Auto-generated SPICE netlist (NO TEXT OCR USED)

Q1 n5 n9 n11 QNPNdefault
Q2 n1 n3 0 QNPNdefault
Q3 0 n8 n5 QPNPdefault
R1 n5 0 1k
R2 0 n7 1k
R3 n10 n8 1k
C1 0 n2 1u
R4 0 n9 1k
R5 n9 n8 1k
R6 n1 n3 1k
R7 n3 n4 1k
R8 n11 n8 1k
R9 n11 n8 1k
R10 0 n5 1k
C2 0 n2 1u
C3 n3 n4 1u
C4 0 n10 1u
* SAME_NODE_SKIPPED Capacitor both_on=n1
V1 n2 0 DC 5
C5 n8 n9 1u
I1 n8 n6 DC 1m
C6 n7 n6 1u
R11 n9 n8 1k
.model QNPNdefault NPN
.model QPNPdefault PNP

.op
.end

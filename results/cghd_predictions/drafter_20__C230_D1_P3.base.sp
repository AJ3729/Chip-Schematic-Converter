* Auto-generated SPICE netlist (NO TEXT OCR USED)

R1 n3 n5 1k
Q1 n1 n3 n4 QNPNdefault
R2 n6 n4 1k
R3 n4 n7 1k
Q2 0 n6 n4 QPNPdefault
R4 n12 0 1k
R5 n13 0 1k
C1 n14 0 1u
C2 n3 n5 1u
C3 n4 n11 1u
R6 n9 n14 1k
R7 n4 0 1k
C4 n10 0 1u
C5 n1 n2 1u
R8 n1 n3 1k
R9 n4 n6 1k
R10 n11 0 1k
R11 n9 0 1k
C6 n1 n2 1u
I1 n10 0 DC 1m
V1 n2 0 DC 5
V2 n4 0 DC 5
I2 0 n8 DC 1m
Q3 n6 0 n9 QNPNdefault
* SAME_NODE_SKIPPED Capacitor both_on=n4
* SAME_NODE_SKIPPED V-DC (one port) both_on=0
Q4 n6 0 n9 QNPNdefault
C7 n7 n8 1u
* UNSNAPPED Diode raw_nodes=[7, None]
.model QNPNdefault NPN
.model QPNPdefault PNP

.op
.end

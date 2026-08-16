* Auto-generated SPICE netlist (NO TEXT OCR USED)

E1 n15 0 n1 n13 100k
E2 n11 0 n1 n12 100k
E3 n5 0 n1 n6 100k
C1 n1 n12 1u
C2 n1 n6 1u
C3 n12 n11 1u
* SAME_NODE_SKIPPED Capacitor both_on=n1
C4 n11 n1 1u
C5 n14 n15 1u
C6 n1 n13 1u
C7 n6 n1 1u
C8 n13 n14 1u
C9 n7 n1 1u
Q1 n2 n3 n4 QNPNdefault
C10 n15 n1 1u
C11 n12 n11 1u
I1 n10 n1 AC 1m
D1 n5 n3 Ddefault
Q2 n8 n9 n1 QPNPdefault
V1 n1 0 DC 5
C12 n4 n1 1u
L1 n12 n1 1m
* SAME_NODE_SKIPPED Inductor both_on=0
C13 n6 n1 1u
L2 n2 n3 1m
R1 n5 n8 1k
C14 n13 n15 1u
.model Ddefault D
.model QNPNdefault NPN
.model QPNPdefault PNP

.op
.end

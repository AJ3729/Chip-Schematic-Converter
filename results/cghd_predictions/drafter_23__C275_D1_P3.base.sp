* Auto-generated SPICE netlist (NO TEXT OCR USED)

E1 n6 0 n4 n4 100k
E2 n16 0 0 n15 100k
E3 n13 0 n14 0 100k
C1 n14 0 1u
C2 n14 n13 1u
C3 0 n4 1u
C4 n7 n4 1u
* SAME_NODE_SKIPPED Capacitor both_on=0
C5 n13 0 1u
V1 n4 0 DC 5
D1 n6 n1 Ddefault
C6 n5 n8 1u
C7 0 n15 1u
Q1 n9 n4 n10 QNPNdefault
R1 n6 n9 1k
V2 n11 0 DC 5
Q2 n4 n1 n3 QNPNdefault
C8 n16 0 1u
C9 n15 n16 1u
I1 n12 0 AC 1m
C10 n14 n13 1u
C11 n16 0 1u
V3 n4 0 DC 5
Q3 n4 0 n2 QPNPdefault
.model Ddefault D
.model QNPNdefault NPN
.model QPNPdefault PNP

.op
.end

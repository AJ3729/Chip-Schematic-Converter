* Auto-generated SPICE netlist (NO TEXT OCR USED)

D1 n5 n3 Ddefault
L1 0 n3 1m
L2 0 n3 1m
C1 n3 n8 1u
L3 n1 n5 1m
C2 n3 n8 1u
D2 n5 n10 Ddefault
* SAME_NODE_SKIPPED Diode both_on=n1
Q1 n1 n1 n1 QNPNdefault
D3 n6 n5 Zdefault
V1 0 n2 DC 5
L4 n2 n3 1m
C3 n1 n3 1u
D4 n4 n1 Ddefault
D5 n10 n3 Ddefault
C4 n3 n8 1u
Q2 n3 n6 n6 QPNPdefault
D6 n1 n3 Zdefault
Q3 n9 n10 n3 QNPNdefault
Q4 n4 n1 n1 QNPNdefault
D7 n3 n1 Ddefault
Q5 n3 n3 n3 QNPNdefault
R1 n3 n7 1k
C5 n4 n3 1u
.model Ddefault D
.model QNPNdefault NPN
.model QPNPdefault PNP
.model Zdefault D(bv=5.1)

.op
.end

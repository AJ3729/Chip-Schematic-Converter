* Auto-generated SPICE netlist (NO TEXT OCR USED)

D1 n2 n6 Ddefault
L1 n1 n2 1m
* SAME_NODE_SKIPPED Diode both_on=n2
C1 n2 n7 1u
D2 n2 n1 Ddefault
C2 n2 n7 1u
L2 n4 n2 1m
L3 n3 n2 1m
C3 n2 n7 1u
L4 n5 n2 1m
D3 n2 n8 Ddefault
C4 n1 n2 1u
* SAME_NODE_SKIPPED Diode both_on=n2
D4 n1 n2 Ddefault
I1 n3 0 DC 1m
Q1 n2 n6 n8 QNPNdefault
V1 0 n5 AC 1
V2 0 n4 AC 1
Q2 n2 n2 n2 QNPNdefault
.model Ddefault D
.model QNPNdefault NPN

.op
.end

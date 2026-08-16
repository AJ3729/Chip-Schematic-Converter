* Auto-generated SPICE netlist (NO TEXT OCR USED)

L1 n3 n6 1m
C1 n5 n10 1u
I1 n7 0 DC 1m
I2 0 n9 DC 1m
C2 n5 n10 1u
L2 n7 n5 1m
D1 n2 n5 Ddefault
D2 n5 n2 Ddefault
D3 n5 n6 Ddefault
D4 n5 n6 Ddefault
L3 n8 n5 1m
L4 n9 n5 1m
C3 n5 n10 1u
R1 n5 n6 1k
I3 0 n8 DC 1m
Q1 n1 n3 n2 QPNPdefault
D5 n4 n5 Ddefault
* SAME_NODE_SKIPPED Diode both_on=n5
* SAME_NODE_SKIPPED Diode both_on=n5
Q2 n2 n4 n2 QPNPdefault
.model Ddefault D
.model QPNPdefault PNP

.op
.end

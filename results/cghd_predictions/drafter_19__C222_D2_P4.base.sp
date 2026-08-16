* Auto-generated SPICE netlist (NO TEXT OCR USED)

C1 n5 n11 1u
* SAME_NODE_SKIPPED Diode both_on=n5
* SAME_NODE_SKIPPED Diode both_on=n5
* SAME_NODE_SKIPPED Diode both_on=n5
L1 n2 n8 1m
L2 n6 n5 1m
* SAME_NODE_SKIPPED Diode both_on=n5
I1 n10 0 DC 1m
D1 n3 n1 Ddefault
I2 n6 n7 DC 1m
L3 n10 n5 1m
L4 n9 n5 1m
C2 n5 n11 1u
D2 n2 n1 Ddefault
I3 0 n9 DC 1m
C3 n5 n11 1u
D3 n1 n5 Ddefault
D4 n1 n5 Ddefault
Q1 n1 n3 n3 QNPNdefault
D5 n5 n3 Ddefault
D6 n1 n4 Zdefault
D7 n5 n3 Ddefault
D8 n5 n4 Ddefault
.model Ddefault D
.model QNPNdefault NPN
.model Zdefault D(bv=5.1)

.op
.end

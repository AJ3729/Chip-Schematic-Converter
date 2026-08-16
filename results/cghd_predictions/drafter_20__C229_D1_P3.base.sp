* Auto-generated SPICE netlist (NO TEXT OCR USED)

R1 n1 0 1k
R2 n5 n9 1k
R3 0 n5 1k
R4 n5 n8 1k
D1 n7 0 Ddefault
D2 n3 n6 Zdefault
D3 n7 0 Ddefault
R5 n2 n3 1k
* SAME_NODE_SKIPPED Zener Diode both_on=0
Q1 n8 n9 n7 QNPNdefault
R6 n4 n6 1k
D4 n4 n3 Zdefault
Q2 n5 0 n3 QPNPdefault
.model Ddefault D
.model QNPNdefault NPN
.model QPNPdefault PNP
.model Zdefault D(bv=5.1)

.op
.end

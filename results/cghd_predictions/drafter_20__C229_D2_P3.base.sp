* Auto-generated SPICE netlist (NO TEXT OCR USED)

Q1 n3 n2 n4 QNPNdefault
D1 0 n2 Ddefault
Q2 n5 n6 0 QNPNdefault
D2 0 n2 Ddefault
R1 n1 n3 1k
R2 n2 n5 1k
* SAME_NODE_SKIPPED Resistor both_on=n2
R3 n1 n2 1k
R4 n2 n6 1k
Q3 n2 n3 n3 QNPNdefault
R5 n2 n4 1k
R6 n1 n2 1k
D3 n4 n2 Zdefault
R7 n1 n2 1k
Q4 0 n2 n4 QNPNdefault
.model Ddefault D
.model QNPNdefault NPN
.model Zdefault D(bv=5.1)

.op
.end

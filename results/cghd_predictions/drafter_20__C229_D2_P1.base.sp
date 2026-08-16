* Auto-generated SPICE netlist (NO TEXT OCR USED)

* SAME_NODE_SKIPPED Diode both_on=0
D1 n5 0 Ddefault
* SAME_NODE_SKIPPED Diode both_on=0
R1 n1 0 1k
D2 n5 0 Ddefault
R2 n1 0 1k
D3 0 n5 Zdefault
R3 0 n3 1k
R4 n1 0 1k
R5 0 n4 1k
Q1 0 0 0 QNPNdefault
D4 0 n5 Zdefault
* SAME_NODE_SKIPPED Resistor both_on=0
R6 0 n2 1k
Q2 n3 n4 0 QNPNdefault
V1 n1 0 DC 5
* SAME_NODE_SKIPPED V-DC (one port) both_on=0
.model Ddefault D
.model QNPNdefault NPN
.model Zdefault D(bv=5.1)

.op
.end

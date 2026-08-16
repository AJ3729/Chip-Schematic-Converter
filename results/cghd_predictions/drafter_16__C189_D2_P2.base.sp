* Auto-generated SPICE netlist (NO TEXT OCR USED)

E1 n1 0 n1 n10 100k
D1 n2 n1 Zdefault
* SAME_NODE_SKIPPED Resistor both_on=n1
R1 n2 n1 1k
C1 n8 n1 1u
E2 n1 0 n1 n1 100k
* SAME_NODE_SKIPPED Capacitor both_on=n1
R2 n9 0 1k
R3 n1 n4 1k
R4 n7 n8 1k
* SAME_NODE_SKIPPED Resistor both_on=n1
C2 n8 n5 1u
C3 n3 n5 1u
R5 n4 n5 1k
R6 n6 n8 1k
M1 n3 n5 n8 n8 NMOSdefault
Q1 n8 n5 n1 QNPNdefault
R7 n3 n5 1k
C4 n9 n1 1u
* SAME_NODE_SKIPPED Resistor both_on=n1
.model NMOSdefault NMOS
.model QNPNdefault NPN
.model Zdefault D(bv=5.1)

.op
.end

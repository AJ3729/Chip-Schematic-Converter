* Auto-generated SPICE netlist (NO TEXT OCR USED)

* SAME_NODE_SKIPPED Zener Diode both_on=n3
E1 n3 0 n11 n3 100k
E2 n10 0 n3 n3 100k
C1 n8 n3 1u
R1 n1 n3 1k
* SAME_NODE_SKIPPED Capacitor both_on=n3
C2 n8 n7 1u
R2 n9 0 1k
D1 n1 n3 Zdefault
* SAME_NODE_SKIPPED Resistor both_on=n3
R3 n2 n5 1k
* SAME_NODE_SKIPPED Resistor both_on=n3
R4 n5 n7 1k
R5 n6 n7 1k
C3 n9 n3 1u
Q1 n3 n5 n4 QNPNdefault
.model QNPNdefault NPN
.model Zdefault D(bv=5.1)

.op
.end

* Auto-generated SPICE netlist (NO TEXT OCR USED)

C1 n13 n14 1u
C2 n16 n17 1u
R1 n21 n15 1k
C3 n7 n6 1u
C4 n15 n1 1u
C5 n6 n19 1u
R2 n12 n20 1k
* SAME_NODE_SKIPPED Capacitor both_on=n1
C6 n5 n4 1u
R3 n19 n18 1k
R4 n5 n4 1k
R5 n18 n23 1k
E1 n6 0 0 n7 100k
E2 n13 0 n5 n20 100k
R6 n9 n1 1k
Q1 n1 n11 n24 QNPNdefault
C7 n2 n3 1u
D1 n8 n10 Zdefault
* SAME_NODE_SKIPPED Resistor both_on=n2
C8 n14 n15 1u
R7 n8 n10 1k
L1 n1 n7 1m
.model QNPNdefault NPN
.model Zdefault D(bv=5.1)

.op
.end

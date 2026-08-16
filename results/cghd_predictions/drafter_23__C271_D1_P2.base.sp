* Auto-generated SPICE netlist (NO TEXT OCR USED)

C1 0 n2 1u
C2 n7 0 1u
* SAME_NODE_SKIPPED Capacitor both_on=0
R1 n8 0 1k
R2 0 n9 1k
D1 n6 0 Zdefault
R3 0 n8 1k
* SAME_NODE_SKIPPED Resistor both_on=0
C3 0 n1 1u
I1 0 n10 AC 1m
R4 n4 0 1k
* SAME_NODE_SKIPPED Resistor both_on=0
R5 n2 n3 1k
R6 n6 0 1k
R7 0 n7 1k
I2 n9 0 AC 1m
* SAME_NODE_SKIPPED Capacitor both_on=0
R8 n4 0 1k
L1 0 n5 1m
R9 0 n4 1k
V1 n10 0 DC 5
L2 0 n7 1m
E1 0 0 0 0 100k
.model Zdefault D(bv=5.1)

.op
.end

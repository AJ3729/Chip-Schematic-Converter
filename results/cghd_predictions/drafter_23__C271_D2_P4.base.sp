* Auto-generated SPICE netlist (NO TEXT OCR USED)

I1 n6 0 AC 1m
R1 n4 0 1k
C1 0 n1 1u
* SAME_NODE_SKIPPED Capacitor both_on=0
D1 n1 0 Ddefault
* SAME_NODE_SKIPPED Resistor both_on=0
* SAME_NODE_SKIPPED Resistor both_on=0
E1 0 0 0 0 100k
I2 0 n2 AC 1m
R2 0 n4 1k
C2 n7 0 1u
R3 n3 n6 1k
C3 n3 0 1u
R4 n5 0 1k
R5 0 n7 1k
* SAME_NODE_SKIPPED Resistor both_on=0
R6 0 n5 1k
L1 0 n4 1m
.model Ddefault D

.op
.end

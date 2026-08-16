* Auto-generated SPICE netlist (NO TEXT OCR USED)

E1 0 0 0 0 100k
E2 0 0 0 0 100k
* SAME_NODE_SKIPPED Capacitor both_on=0
R1 0 n3 1k
C1 0 n2 1u
C2 n5 n1 1u
C3 n8 0 1u
R2 0 n6 1k
R3 n1 0 1k
E3 0 0 0 0 100k
R4 n7 0 1k
* SAME_NODE_SKIPPED Resistor both_on=0
I1 n9 0 AC 1m
R5 0 n5 1k
R6 n3 0 1k
R7 0 n9 1k
R8 0 n8 1k
L1 n6 0 1m
I2 0 n4 AC 1m
V1 0 n4 DC 5

.op
.end

* Auto-generated SPICE netlist (NO TEXT OCR USED)

I1 n10 0 AC 1m
E1 n7 0 0 0 100k
I2 0 n5 AC 1m
C1 n6 n1 1u
E2 0 0 0 0 100k
R1 0 n4 1k
R2 0 n6 1k
* SAME_NODE_SKIPPED Capacitor both_on=0
C2 0 n2 1u
R3 0 n6 1k
C3 n9 n11 1u
R4 n4 0 1k
R5 0 n10 1k
R6 n1 0 1k
R7 0 n9 1k
* SAME_NODE_SKIPPED Resistor both_on=0
R8 n8 0 1k
D1 n3 n2 Ddefault
.model Ddefault D

.op
.end

* Auto-generated SPICE netlist (NO TEXT OCR USED)

I1 0 n1 AC 1m
* SAME_NODE_SKIPPED Capacitor both_on=0
I2 n4 0 AC 1m
E1 0 0 0 0 100k
C1 0 n3 1u
* SAME_NODE_SKIPPED Resistor both_on=0
C2 n2 0 1u
* SAME_NODE_SKIPPED Resistor both_on=0
C3 n5 0 1u
R1 n2 n4 1k
* SAME_NODE_SKIPPED Resistor both_on=0
* SAME_NODE_SKIPPED Resistor both_on=0
* SAME_NODE_SKIPPED Resistor both_on=0
* SAME_NODE_SKIPPED Resistor both_on=0
R2 0 n5 1k
* SAME_NODE_SKIPPED Resistor both_on=0
D1 0 n3 Ddefault
.model Ddefault D

.op
.end

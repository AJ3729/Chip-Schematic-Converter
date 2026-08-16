* Auto-generated SPICE netlist (NO TEXT OCR USED)

I1 0 n6 AC 1m
I2 n5 0 AC 1m
* SAME_NODE_SKIPPED Capacitor both_on=0
R1 n1 0 1k
R2 0 n3 1k
C1 n2 0 1u
* SAME_NODE_SKIPPED Resistor both_on=0
R3 n4 n5 1k
* SAME_NODE_SKIPPED Capacitor both_on=0
* SAME_NODE_SKIPPED Resistor both_on=0
R4 0 n2 1k
R5 n3 0 1k
* SAME_NODE_SKIPPED Resistor both_on=0
C2 0 n4 1u
* SAME_NODE_SKIPPED Zener Diode both_on=0
* SAME_NODE_SKIPPED Resistor both_on=0
* SAME_NODE_SKIPPED Capacitor both_on=0
C3 0 n1 1u
* SAME_NODE_SKIPPED Zener Diode both_on=0
D1 n6 n4 Ddefault
.model Ddefault D

.op
.end

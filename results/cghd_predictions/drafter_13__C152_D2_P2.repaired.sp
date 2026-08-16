* Auto-generated SPICE netlist (NO TEXT OCR USED)

C1 n1 0 1u
C2 0 n3 1u
L1 n2 0 1m
* SAME_NODE_SKIPPED Capacitor both_on=0
V1 n3 0 DC 5
I1 n2 0 DC 1m
L2 n1 n2 1m
V2 0 n2 DC 5
* SAME_NODE_SKIPPED Resistor both_on=0
I2 n2 0 AC 1m

.op
.end

* Auto-generated SPICE netlist (NO TEXT OCR USED)

R1 0 n1 1k
* SAME_NODE_SKIPPED Resistor both_on=n1
R2 n1 n4 1k
R3 0 n2 1k
I1 0 n3 AC 1m
I2 n2 n4 AC 1m
I3 0 n3 DC 1m
D1 n2 n3 Ddefault
.model Ddefault D

.op
.end

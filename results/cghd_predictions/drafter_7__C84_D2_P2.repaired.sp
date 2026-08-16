* Auto-generated SPICE netlist (NO TEXT OCR USED)

R1 0 n2 1k
* SAME_NODE_SKIPPED Resistor both_on=0
R2 n1 n2 1k
R3 0 n1 1k
R4 n2 0 1k
I1 0 n2 DC 1m
V1 n1 0 DC 5
* SAME_NODE_SKIPPED I-AC both_on=0
I2 0 n2 AC 1m

.op
.end

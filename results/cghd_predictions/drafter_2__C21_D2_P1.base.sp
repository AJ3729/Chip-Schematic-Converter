* Auto-generated SPICE netlist (NO TEXT OCR USED)

I1 0 n1 DC 1m
R1 0 n2 1k
R2 0 n9 1k
C1 n3 0 1u
* SAME_NODE_SKIPPED Capacitor both_on=0
V1 n8 0 DC 5
I2 n5 0 DC 1m
M1 0 n4 n7 n7 PMOSdefault
M2 0 n4 n7 n7 PMOSdefault
E1 n6 0 n3 0 100k
* SAME_NODE_SKIPPED Resistor both_on=0
V2 n8 0 DC 5
.model PMOSdefault PMOS

.op
.end

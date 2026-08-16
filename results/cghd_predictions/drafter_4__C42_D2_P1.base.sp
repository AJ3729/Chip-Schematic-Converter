* Auto-generated SPICE netlist (NO TEXT OCR USED)

E1 n4 0 n5 n2 100k
D1 n4 n1 Ddefault
* SAME_NODE_SKIPPED Resistor both_on=0
R1 0 n2 1k
R2 n5 0 1k
R3 n2 0 1k
R4 n1 0 1k
L1 0 n3 1m
E2 0 0 0 0 100k
C1 0 n3 1u
.model Ddefault D

.op
.end

* Auto-generated SPICE netlist (NO TEXT OCR USED)

R1 n5 0 1k
* SAME_NODE_SKIPPED Capacitor both_on=0
R2 n1 0 1k
R3 n4 0 1k
R4 0 n2 1k
R5 0 n1 1k
D1 n3 n2 Ddefault
E1 0 0 0 0 100k
E2 n3 0 n1 n4 100k
.model Ddefault D

.op
.end

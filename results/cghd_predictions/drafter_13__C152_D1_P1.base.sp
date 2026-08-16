* Auto-generated SPICE netlist (NO TEXT OCR USED)

I1 n3 0 AC 1m
L1 n1 n3 1m
L2 n3 0 1m
C1 n1 0 1u
I2 n2 0 DC 1m
C2 n2 0 1u
D1 0 n2 Ddefault
* SAME_NODE_SKIPPED Capacitor both_on=0
.model Ddefault D

.op
.end

* Auto-generated SPICE netlist (NO TEXT OCR USED)

I1 n8 n9 AC 1m
I2 n9 n12 DC 1m
C1 n10 n11 1u
I3 n6 n8 AC 1m
I4 n7 0 DC 1m
I5 n5 n4 DC 1m
M1 n3 n1 n2 n2 NMOSdefault
I6 n2 n3 AC 1m
* SAME_NODE_SKIPPED Capacitor both_on=n2
L1 n2 n3 1m
.model NMOSdefault NMOS

.op
.end

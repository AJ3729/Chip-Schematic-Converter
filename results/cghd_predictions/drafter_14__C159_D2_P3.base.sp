* Auto-generated SPICE netlist (NO TEXT OCR USED)

I1 n4 n6 AC 1m
I2 n6 n7 AC 1m
E1 n5 0 n8 n2 100k
E2 n2 0 n3 n1 100k
I3 n9 n11 AC 1m
C1 n5 n10 1u
I4 n7 n9 AC 1m
Q1 n5 n8 n10 QNPNdefault
I5 n11 n9 DC 1m
M1 n5 n1 n5 n5 NMOSdefault
* UNSNAPPED I-DC raw_nodes=[4, None]
.model NMOSdefault NMOS
.model QNPNdefault NPN

.op
.end

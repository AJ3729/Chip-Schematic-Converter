* Auto-generated SPICE netlist (NO TEXT OCR USED)

V1 0 n2 AC 1
L1 n3 n1 1m
L2 n1 0 1m
C1 n1 0 1u
* SAME_NODE_SKIPPED Capacitor both_on=0
M1 0 0 n4 n4 NMOSdefault
C2 0 n3 1u
.model NMOSdefault NMOS

.op
.end

* Auto-generated SPICE netlist (NO TEXT OCR USED)

E1 n7 0 0 n8 100k
* SAME_NODE_SKIPPED Capacitor both_on=0
* SAME_NODE_SKIPPED Capacitor both_on=0
C1 n2 n3 1u
C2 n4 0 1u
* SAME_NODE_SKIPPED Capacitor both_on=0
M1 n9 0 0 0 NMOSdefault
C3 n1 n2 1u
E2 n5 0 n1 0 100k
M2 0 0 0 0 NMOSdefault
M3 0 0 0 0 NMOSdefault
C4 0 n9 1u
* SAME_NODE_SKIPPED Capacitor both_on=0
C5 0 n6 1u
.model NMOSdefault NMOS

.op
.end

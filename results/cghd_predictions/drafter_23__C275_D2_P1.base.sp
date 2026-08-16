* Auto-generated SPICE netlist (NO TEXT OCR USED)

E1 0 0 n8 0 100k
E2 n1 0 0 n2 100k
C1 n6 0 1u
C2 n10 0 1u
C3 n11 0 1u
C4 n1 0 1u
C5 n8 0 1u
C6 n6 n7 1u
I1 n12 0 AC 1m
* UNSNAPPED Capacitor raw_nodes=[3, None]
* SAME_NODE_SKIPPED Capacitor both_on=0
L1 0 n4 1m
C7 0 n6 1u
C8 n13 0 1u
L2 n1 n2 1m
E3 n6 0 n10 n9 100k
C9 0 n2 1u
M1 0 n5 n4 n4 NMOSdefault
C10 n10 0 1u
C11 n3 0 1u
C12 n8 0 1u
.model NMOSdefault NMOS

.op
.end

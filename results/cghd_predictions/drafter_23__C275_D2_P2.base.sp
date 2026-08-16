* Auto-generated SPICE netlist (NO TEXT OCR USED)

C1 n8 0 1u
C2 n10 0 1u
E1 0 0 0 n7 100k
C3 n8 0 1u
E2 n4 0 0 n3 100k
* SAME_NODE_SKIPPED Capacitor both_on=0
E3 n8 0 0 n9 100k
C4 n1 0 1u
C5 0 n2 1u
L1 n5 0 1m
R1 0 n6 1k
C6 0 n8 1u
* UNSNAPPED Op-Amp raw_nodes=[1, 2, None]
M1 n8 n8 n8 n8 PMOSdefault
.model PMOSdefault PMOS

.op
.end

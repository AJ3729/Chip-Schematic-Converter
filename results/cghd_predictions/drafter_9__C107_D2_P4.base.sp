* Auto-generated SPICE netlist (NO TEXT OCR USED)

E1 n2 0 n3 0 100k
L1 n1 0 1m
* SAME_NODE_SKIPPED Inductor both_on=0
M1 0 0 0 0 PMOSdefault
L2 0 n2 1m
C1 n2 0 1u
L3 0 n2 1m
* SAME_NODE_SKIPPED Capacitor both_on=0
M2 0 0 0 0 NMOSdefault
* SAME_NODE_SKIPPED Inductor both_on=0
C2 0 n4 1u
* SAME_NODE_SKIPPED Inductor both_on=0
* SAME_NODE_SKIPPED Inductor both_on=0
.model NMOSdefault NMOS
.model PMOSdefault PMOS

.op
.end

* Auto-generated SPICE netlist (NO TEXT OCR USED)

C1 0 n1 1u
E1 n4 0 n5 n3 100k
M1 n3 0 n3 n3 NMOSdefault
M2 n8 n9 0 0 NMOSdefault
M3 0 0 n6 n6 NMOSdefault
M4 n10 0 n9 n9 NMOSdefault
C2 n8 n9 1u
Q1 n10 0 n9 QNPNdefault
E2 0 0 0 0 100k
M5 n3 0 n4 n4 NMOSdefault
C3 n2 0 1u
C4 0 n2 1u
M6 n4 0 n3 n3 PMOSdefault
M7 0 n7 n6 n6 NMOSdefault
M8 n8 n8 0 0 NMOSdefault
* SAME_NODE_SKIPPED Capacitor both_on=0
.model NMOSdefault NMOS
.model PMOSdefault PMOS
.model QNPNdefault NPN

.op
.end

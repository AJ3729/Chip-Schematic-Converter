* Auto-generated SPICE netlist (NO TEXT OCR USED)

M1 n1 n8 n6 n6 PMOSdefault
Q1 n16 n19 n3 QNPNdefault
M2 n11 n10 n3 n3 PMOSdefault
C1 n18 n17 1u
C2 n3 n2 1u
M3 n7 n6 n7 n7 NMOSdefault
C3 n13 n3 1u
C4 n8 n7 1u
M4 n3 n4 0 0 PMOSdefault
M5 n3 n7 n3 n3 PMOSdefault
* SAME_NODE_SKIPPED Capacitor both_on=0
Q2 n12 n23 n12 QNPNdefault
M6 n4 n9 n11 n11 NMOSdefault
M7 n6 n3 n3 n3 NMOSdefault
V1 n2 0 DC 5
C5 n11 0 1u
C6 n17 n20 1u
* UNSNAPPED Capacitor raw_nodes=[None, None]
M8 n14 n17 n15 n15 PMOSdefault
M9 0 n12 n22 n22 NMOSdefault
M10 n4 0 n3 n3 NMOSdefault
C7 n4 0 1u
M11 n3 n4 n5 n5 PMOSdefault
* SAME_NODE_SKIPPED V-DC (one port) both_on=0
.model NMOSdefault NMOS
.model PMOSdefault PMOS
.model QNPNdefault NPN

.op
.end

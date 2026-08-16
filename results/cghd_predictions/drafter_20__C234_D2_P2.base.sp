* Auto-generated SPICE netlist (NO TEXT OCR USED)

M1 n2 n9 n5 n5 NMOSdefault
M2 n16 n15 n13 n13 PMOSdefault
C1 n17 n5 1u
C2 n9 n10 1u
C3 n7 0 1u
M3 n21 0 n19 n19 NMOSdefault
M4 n5 n5 n13 n13 PMOSdefault
M5 n3 n10 n3 n3 NMOSdefault
M6 n6 n11 n4 n4 NMOSdefault
M7 n11 n14 n8 n8 NMOSdefault
* UNSNAPPED MOSFET-P raw_nodes=[9, 15, None]
M8 n10 n5 n10 n10 NMOSdefault
C4 n1 n20 1u
V1 n1 0 DC 5
M9 n8 n11 n4 n4 NMOSdefault
M10 n8 n15 n16 n16 NMOSdefault
C5 n16 0 1u
M11 n20 n23 n26 n26 NMOSdefault
M12 0 n21 n19 n19 NMOSdefault
M13 n23 n22 n18 n18 PMOSdefault
M14 n3 n13 n12 n12 PMOSdefault
M15 n23 n22 n22 n22 PMOSdefault
Q1 0 0 n27 QNPNdefault
M16 n5 n16 n24 n24 PMOSdefault
M17 n16 n8 n15 n15 NMOSdefault
C6 n23 n25 1u
.model NMOSdefault NMOS
.model PMOSdefault PMOS
.model QNPNdefault NPN

.op
.end

* Auto-generated SPICE netlist (NO TEXT OCR USED)

M1 n13 n17 n13 n13 PMOSdefault
M2 n7 n10 n5 n5 NMOSdefault
M3 n4 n12 n9 n9 NMOSdefault
M4 n9 n19 n9 n9 PMOSdefault
M5 n3 n9 n3 n3 PMOSdefault
C1 n6 n16 1u
C2 n20 n19 1u
C3 n1 n19 1u
C4 n11 n10 1u
C5 n17 n16 1u
M6 n17 n25 n27 n27 PMOSdefault
M7 n9 n9 n9 n9 PMOSdefault
M8 n7 n17 n15 n15 NMOSdefault
C6 n12 n9 1u
M9 n24 n22 n17 n17 NMOSdefault
M10 n7 n14 n11 n11 NMOSdefault
* SAME_NODE_SKIPPED Capacitor both_on=n26
M11 n19 n18 n13 n13 PMOSdefault
V1 n32 0 DC 5
V2 n13 0 DC 5
M12 n23 n26 n21 n21 PMOSdefault
C7 n24 n30 1u
E1 n2 0 n2 n2 100k
V3 n32 0 DC 5
M13 n31 n26 n26 n26 PMOSdefault
Q1 n19 n25 n28 QNPNdefault
C8 n15 n17 1u
C9 n26 n29 1u
M14 n26 n26 n31 n31 PMOSdefault
M15 n5 n8 n11 n11 PMOSdefault
Q2 n24 0 n24 QNPNdefault
.model NMOSdefault NMOS
.model PMOSdefault PMOS
.model QNPNdefault NPN

* --- design-intent repair (does not change topology) ---
Rshunt_n1 n1 0 1e+09
Rshunt_n2 n2 0 1e+09
Rshunt_n21 n21 0 1e+09
Rshunt_n32 n32 0 1e+09

.op
.end

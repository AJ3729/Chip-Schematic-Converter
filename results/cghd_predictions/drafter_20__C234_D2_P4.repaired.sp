* Auto-generated SPICE netlist (NO TEXT OCR USED)

M1 n14 n13 n11 n11 PMOSdefault
M2 n6 n6 n9 n9 NMOSdefault
* SAME_NODE_SKIPPED Capacitor both_on=0
M3 n6 0 n5 n5 NMOSdefault
M4 n8 n8 n8 n8 PMOSdefault
M5 n12 n14 n6 n6 NMOSdefault
M6 0 n19 n14 n14 NMOSdefault
C1 n17 n16 1u
I1 n24 n25 AC 1m
M7 n10 n11 n2 n2 PMOSdefault
M8 n4 n4 n8 n8 PMOSdefault
C2 n3 n8 1u
C3 n18 n22 1u
C4 n9 0 1u
C5 n21 n18 1u
M9 n9 n7 n5 n5 PMOSdefault
M10 n16 n15 n11 n11 PMOSdefault
M11 n8 n16 n8 n8 PMOSdefault
M12 n8 n8 n8 n8 NMOSdefault
C6 n24 n25 1u
C7 n14 0 1u
M13 n14 n16 n20 n20 PMOSdefault
M14 0 n23 0 0 NMOSdefault
C8 n16 n14 1u
.model NMOSdefault NMOS
.model PMOSdefault PMOS

* --- design-intent repair (does not change topology) ---
Rshunt_n1 n1 0 1e+09
Rshunt_n10 n10 0 1e+09
Rshunt_n12 n12 0 1e+09
Rshunt_n13 n13 0 1e+09
Rshunt_n15 n15 0 1e+09
Rshunt_n17 n17 0 1e+09
Rshunt_n18 n18 0 1e+09
Rshunt_n19 n19 0 1e+09
Rshunt_n23 n23 0 1e+09
Rshunt_n24 n24 0 1e+09
Rshunt_n3 n3 0 1e+09

.op
.end

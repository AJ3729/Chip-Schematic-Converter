* Auto-generated SPICE netlist (NO TEXT OCR USED)

C1 0 n7 1u
I1 n3 n2 AC 1m
M1 n7 n4 n5 n5 NMOSdefault
M2 n5 n8 n6 n6 NMOSdefault
M3 n5 n9 n5 n5 PMOSdefault
* SAME_NODE_SKIPPED Capacitor both_on=0
M4 n12 n13 n13 n13 NMOSdefault
* SAME_NODE_SKIPPED Capacitor both_on=n6
M5 n7 n7 n7 n7 NMOSdefault
C2 0 n14 1u
M6 n11 n11 n11 n11 NMOSdefault
C3 n13 n6 1u
C4 n5 n8 1u
C5 0 n7 1u
M7 0 n7 n4 n4 NMOSdefault
Q1 n5 n7 n15 QNPNdefault
M8 n7 n10 n5 n5 PMOSdefault
C6 n5 n6 1u
I2 n3 n1 AC 1m
M9 n4 0 n7 n7 PMOSdefault
.model NMOSdefault NMOS
.model PMOSdefault PMOS
.model QNPNdefault NPN

* --- design-intent repair (does not change topology) ---
Rshunt_n1 n1 0 1e+09
Rshunt_n10 n10 0 1e+09
Rshunt_n11 n11 0 1e+09
Rshunt_n12 n12 0 1e+09
Rshunt_n14 n14 0 1e+09
Rshunt_n8 n8 0 1e+09
Rshunt_n9 n9 0 1e+09

.op
.end

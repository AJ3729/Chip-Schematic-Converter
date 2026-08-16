* Auto-generated SPICE netlist (NO TEXT OCR USED)

M1 n9 n9 n9 n9 NMOSdefault
M2 n1 n5 n2 n2 PMOSdefault
* SAME_NODE_SKIPPED Capacitor both_on=n1
* SAME_NODE_SKIPPED Capacitor both_on=n1
* SAME_NODE_SKIPPED Capacitor both_on=n1
M3 n1 n1 n1 n1 NMOSdefault
M4 n8 n10 n10 n10 NMOSdefault
C1 n10 n1 1u
C2 n11 n12 1u
* SAME_NODE_SKIPPED Capacitor both_on=n1
M5 n7 n6 n2 n2 PMOSdefault
* SAME_NODE_SKIPPED Capacitor both_on=n1
C3 n4 n1 1u
C4 n3 n4 1u
M6 n1 n4 n13 n13 NMOSdefault
C5 n1 n2 1u
* SAME_NODE_SKIPPED Capacitor both_on=n10
M7 n1 n11 0 0 NMOSdefault
* SAME_NODE_SKIPPED Capacitor both_on=n1
.model NMOSdefault NMOS
.model PMOSdefault PMOS

* --- design-intent repair (does not change topology) ---
Rshunt_n10 n10 0 1e+09
Rshunt_n11 n11 0 1e+09
Rshunt_n3 n3 0 1e+09
Rshunt_n5 n5 0 1e+09
Rshunt_n6 n6 0 1e+09
Rshunt_n9 n9 0 1e+09

.op
.end

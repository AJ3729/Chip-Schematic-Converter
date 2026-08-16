* Auto-generated SPICE netlist (NO TEXT OCR USED)

C1 n1 n9 1u
M1 n3 n4 n2 n2 NMOSdefault
* SAME_NODE_SKIPPED Capacitor both_on=n1
C2 n13 n15 1u
C3 0 n2 1u
M2 n1 n1 n1 n1 NMOSdefault
M3 n10 n10 n10 n10 NMOSdefault
Q1 n2 n12 n16 QNPNdefault
* UNSNAPPED MOSFET-P raw_nodes=[15, 7, None]
M4 n14 n13 n1 n1 PMOSdefault
* SAME_NODE_SKIPPED Capacitor both_on=n1
M5 n11 0 0 0 NMOSdefault
M6 n1 n1 n9 n9 PMOSdefault
M7 n2 n8 n1 n1 PMOSdefault
C4 n3 n5 1u
C5 n3 n4 1u
V1 n12 0 DC 5
* SAME_NODE_SKIPPED Capacitor both_on=0
* SAME_NODE_SKIPPED Capacitor both_on=n2
V2 n6 0 DC 5
M8 n3 n7 n2 n2 PMOSdefault
Q2 n9 n12 n17 QNPNdefault
C6 n1 n3 1u
.model NMOSdefault NMOS
.model PMOSdefault PMOS
.model QNPNdefault NPN

* --- design-intent repair (does not change topology) ---
Rshunt_n1 n1 0 1e+09
Rshunt_n10 n10 0 1e+09
Rshunt_n11 n11 0 1e+09
Rshunt_n6 n6 0 1e+09

.op
.end

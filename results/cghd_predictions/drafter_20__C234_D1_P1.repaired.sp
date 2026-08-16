* Auto-generated SPICE netlist (NO TEXT OCR USED)

M1 n2 n6 n8 n8 NMOSdefault
* UNSNAPPED MOSFET-P raw_nodes=[3, 7, None]
M2 n9 n4 n5 n5 NMOSdefault
C1 n15 n2 1u
C2 n10 n7 1u
M3 n6 n13 n3 n3 PMOSdefault
* SAME_NODE_SKIPPED Capacitor both_on=n10
* SAME_NODE_SKIPPED Capacitor both_on=n2
M4 n7 n10 n4 n4 NMOSdefault
M5 n7 n7 n9 n9 NMOSdefault
C3 n3 n8 1u
C4 n10 n16 1u
M6 n14 n15 n15 n15 NMOSdefault
M7 n10 n7 n16 n16 NMOSdefault
C5 n6 n2 1u
R1 n1 n10 1k
V1 n7 0 DC 5
C6 n10 n9 1u
C7 n11 n6 1u
Q1 n6 n7 0 QNPNdefault
V2 n12 n2 AC 1
* SAME_NODE_SKIPPED Capacitor both_on=n15
.model NMOSdefault NMOS
.model PMOSdefault PMOS
.model QNPNdefault NPN

* --- design-intent repair (does not change topology) ---
Rshunt_n1 n1 0 1e+09
Rshunt_n11 n11 0 1e+09
Rshunt_n12 n12 0 1e+09
Rshunt_n13 n13 0 1e+09

.op
.end

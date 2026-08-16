* Auto-generated SPICE netlist (NO TEXT OCR USED)

E1 n9 0 n11 n5 100k
* UNSNAPPED Capacitor raw_nodes=[21, None]
C1 n10 n6 1u
C2 n2 n6 1u
M1 0 0 0 0 NMOSdefault
E2 0 0 n6 n4 100k
M2 n8 n4 0 0 PMOSdefault
C3 n7 n4 1u
M3 0 n9 n5 n5 NMOSdefault
M4 n4 0 n8 n8 PMOSdefault
C4 n1 n3 1u
M5 n6 0 0 0 PMOSdefault
C5 n5 n7 1u
Q1 n12 n12 n12 QPNPdefault
.model NMOSdefault NMOS
.model PMOSdefault PMOS
.model QPNPdefault PNP

* --- design-intent repair (does not change topology) ---
Rshunt_n1 n1 0 1e+09
Rshunt_n10 n10 0 1e+09
Rshunt_n11 n11 0 1e+09
Rshunt_n12 n12 0 1e+09
Rshunt_n2 n2 0 1e+09
Rshunt_n7 n7 0 1e+09

.op
.end

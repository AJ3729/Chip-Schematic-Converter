* Auto-generated SPICE netlist (NO TEXT OCR USED)

C1 n8 0 1u
C2 n10 0 1u
E1 0 0 0 n7 100k
C3 n8 0 1u
E2 n4 0 0 n3 100k
* SAME_NODE_SKIPPED Capacitor both_on=0
E3 n8 0 0 n9 100k
C4 n1 0 1u
C5 0 n2 1u
L1 n5 0 1m
R1 0 n6 1k
C6 0 n8 1u
* UNSNAPPED Op-Amp raw_nodes=[1, 2, None]
M1 n8 n8 n8 n8 PMOSdefault
.model PMOSdefault PMOS

* --- design-intent repair (does not change topology) ---
Rshunt_n1 n1 0 1e+09
Rshunt_n10 n10 0 1e+09
Rshunt_n2 n2 0 1e+09
Rshunt_n3 n3 0 1e+09
Rshunt_n5 n5 0 1e+09
Rshunt_n6 n6 0 1e+09
Rshunt_n7 n7 0 1e+09
Rshunt_n9 n9 0 1e+09

.op
.end

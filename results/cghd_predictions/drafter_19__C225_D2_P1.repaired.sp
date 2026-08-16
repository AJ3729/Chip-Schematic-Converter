* Auto-generated SPICE netlist (NO TEXT OCR USED)

E1 n9 0 n12 n6 100k
* SAME_NODE_SKIPPED Capacitor both_on=0
C1 n15 0 1u
M1 n6 0 n9 n9 NMOSdefault
M2 n15 0 0 0 NMOSdefault
E2 n10 0 n7 n11 100k
C2 n3 0 1u
C3 n15 0 1u
C4 n5 n3 1u
C5 n1 n2 1u
M3 n14 0 0 0 NMOSdefault
* SAME_NODE_SKIPPED I-AC both_on=0
M4 0 0 0 0 NMOSdefault
C6 0 n14 1u
C7 n13 0 1u
C8 n6 n7 1u
C9 0 n14 1u
C10 n6 n8 1u
C11 n4 n5 1u
M5 n7 0 n10 n10 NMOSdefault
C12 n2 n4 1u
C13 n4 n1 1u
C14 n1 n3 1u
.model NMOSdefault NMOS

* --- design-intent repair (does not change topology) ---
Rshunt_n1 n1 0 1e+09
Rshunt_n10 n10 0 1e+09
Rshunt_n13 n13 0 1e+09

.op
.end

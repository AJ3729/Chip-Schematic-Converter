* Auto-generated SPICE netlist (NO TEXT OCR USED)

L1 0 n4 1m
L2 n5 0 1m
C1 0 n7 1u
L3 0 n7 1m
C2 0 n6 1u
C3 n4 n7 1u
C4 n3 n5 1u
R1 n2 n1 1k

* --- design-intent repair (does not change topology) ---
Rshunt_n1 n1 0 1e+09
Rshunt_n3 n3 0 1e+09
Rshunt_n6 n6 0 1e+09

.op
.end

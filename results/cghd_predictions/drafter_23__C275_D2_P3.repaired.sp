* Auto-generated SPICE netlist (NO TEXT OCR USED)

C1 0 n5 1u
* SAME_NODE_SKIPPED Capacitor both_on=0
E1 0 0 0 0 100k
* SAME_NODE_SKIPPED Capacitor both_on=0
C2 0 n2 1u
C3 n6 0 1u
C4 n4 0 1u
E2 0 0 0 0 100k
C5 n1 0 1u
* SAME_NODE_SKIPPED Inductor both_on=0
L1 n3 0 1m

* --- design-intent repair (does not change topology) ---
Rshunt_n1 n1 0 1e+09
Rshunt_n2 n2 0 1e+09
Rshunt_n3 n3 0 1e+09
Rshunt_n4 n4 0 1e+09
Rshunt_n5 n5 0 1e+09
Rshunt_n6 n6 0 1e+09

.op
.end

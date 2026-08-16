* Auto-generated SPICE netlist (NO TEXT OCR USED)

E1 n3 0 n3 0 100k
* SAME_NODE_SKIPPED V-DC both_on=0
C1 n4 0 1u
* SAME_NODE_SKIPPED Capacitor both_on=0
C2 0 n4 1u
C3 0 n3 1u
E2 n2 0 0 n1 100k
E3 n3 0 0 0 100k

* --- design-intent repair (does not change topology) ---
Rshunt_n1 n1 0 1e+09
Rshunt_n4 n4 0 1e+09

.op
.end

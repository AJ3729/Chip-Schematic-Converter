* Auto-generated SPICE netlist (NO TEXT OCR USED)

* SAME_NODE_SKIPPED Capacitor both_on=0
R1 0 n1 1k
R2 0 n5 1k
R3 n1 0 1k
L1 n6 0 1m
C1 n2 0 1u
C2 n4 0 1u
* SAME_NODE_SKIPPED Inductor both_on=0
C3 n3 0 1u

* --- design-intent repair (does not change topology) ---
Rshunt_n2 n2 0 1e+09
Rshunt_n3 n3 0 1e+09
Rshunt_n4 n4 0 1e+09
Rshunt_n5 n5 0 1e+09
Rshunt_n6 n6 0 1e+09

.op
.end

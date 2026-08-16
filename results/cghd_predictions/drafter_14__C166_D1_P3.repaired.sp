* Auto-generated SPICE netlist (NO TEXT OCR USED)

* SAME_NODE_SKIPPED Inductor both_on=0
C1 n2 n1 1u
* SAME_NODE_SKIPPED Capacitor both_on=0
R1 n4 0 1k
C2 n3 0 1u
C3 0 n3 1u
R2 n5 n6 1k

* --- design-intent repair (does not change topology) ---
Rshunt_n1 n1 0 1e+09
Rshunt_n3 n3 0 1e+09
Rshunt_n4 n4 0 1e+09
Rshunt_n5 n5 0 1e+09

.op
.end

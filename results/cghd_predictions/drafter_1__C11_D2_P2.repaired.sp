* Auto-generated SPICE netlist (NO TEXT OCR USED)

C1 0 n6 1u
* SAME_NODE_SKIPPED Capacitor both_on=0
I1 n7 n5 AC 1m
R1 n3 0 1k
C2 0 n4 1u
R2 n1 n2 1k

* --- design-intent repair (does not change topology) ---
Rshunt_n1 n1 0 1e+09
Rshunt_n3 n3 0 1e+09
Rshunt_n4 n4 0 1e+09
Rshunt_n5 n5 0 1e+09
Rshunt_n6 n6 0 1e+09

.op
.end

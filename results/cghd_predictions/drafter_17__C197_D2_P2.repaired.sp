* Auto-generated SPICE netlist (NO TEXT OCR USED)

I1 n4 n7 AC 1m
C1 n2 n6 1u
C2 n3 n8 1u
R1 n3 n10 1k
C3 0 n5 1u
* SAME_NODE_SKIPPED Inductor both_on=n2
* SAME_NODE_SKIPPED Capacitor both_on=n1
R2 n6 n9 1k

* --- design-intent repair (does not change topology) ---
Rshunt_n1 n1 0 1e+09
Rshunt_n10 n10 0 1e+09
Rshunt_n2 n2 0 1e+09
Rshunt_n4 n4 0 1e+09
Rshunt_n5 n5 0 1e+09

.op
.end

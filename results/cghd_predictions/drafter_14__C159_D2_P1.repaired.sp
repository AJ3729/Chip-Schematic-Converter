* Auto-generated SPICE netlist (NO TEXT OCR USED)

I1 n3 n5 AC 1m
E1 0 0 n1 n2 100k
I2 n8 n9 AC 1m
C1 n7 n1 1u
I3 n5 n6 AC 1m
I4 n6 n8 AC 1m
* SAME_NODE_SKIPPED Capacitor both_on=n1
V1 n1 0 AC 1
E2 n7 0 n1 0 100k

* --- design-intent repair (does not change topology) ---
Rshunt_n2 n2 0 1e+09
Rshunt_n3 n3 0 1e+09
Rshunt_n4 n4 0 1e+09

.op
.end

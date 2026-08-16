* Auto-generated SPICE netlist (NO TEXT OCR USED)

I1 n8 n5 DC 1m
C1 n6 n7 1u
* SAME_NODE_SKIPPED Capacitor both_on=n1
I2 n4 n5 AC 1m
I3 n2 n1 DC 1m
I4 n3 n4 AC 1m
E1 n6 0 n1 n1 100k
V1 n1 n2 AC 1

* --- design-intent repair (does not change topology) ---
Rshunt_n1 n1 0 1e+09
Rshunt_n3 n3 0 1e+09

.op
.end

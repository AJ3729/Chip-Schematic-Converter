* Auto-generated SPICE netlist (NO TEXT OCR USED)

I1 n7 n6 DC 1m
* SAME_NODE_SKIPPED Inductor both_on=n2
C1 n2 n3 1u
L1 n2 n4 1m
I2 n7 n5 DC 1m
C2 n2 n3 1u
C3 n2 n3 1u
L2 n2 n5 1m
L3 n2 n5 1m
I3 n4 n7 DC 1m
L4 n2 n6 1m

* --- design-intent repair (does not change topology) ---
Rshunt_n1 n1 0 1e+09
Rshunt_n2 n2 0 1e+09

.op
.end

* Auto-generated SPICE netlist (NO TEXT OCR USED)

I1 n8 n9 AC 1m
I2 n9 n12 DC 1m
C1 n10 n11 1u
I3 n6 n8 AC 1m
I4 n7 0 DC 1m
I5 n5 n4 DC 1m
M1 n3 n1 n2 n2 NMOSdefault
I6 n2 n3 AC 1m
* SAME_NODE_SKIPPED Capacitor both_on=n2
L1 n2 n3 1m
.model NMOSdefault NMOS

* --- design-intent repair (does not change topology) ---
Rshunt_n1 n1 0 1e+09
Rshunt_n10 n10 0 1e+09
Rshunt_n12 n12 0 1e+09
Rshunt_n4 n4 0 1e+09
Rshunt_n7 n7 0 1e+09

.op
.end

* Auto-generated SPICE netlist (NO TEXT OCR USED)

* SAME_NODE_SKIPPED Capacitor both_on=0
C1 0 n3 1u
I1 n1 n3 DC 1m
R1 0 n1 1k
L1 0 n2 1m
* SAME_NODE_SKIPPED Capacitor both_on=0

* --- design-intent repair (does not change topology) ---
Rshunt_n2 n2 0 1e+09
Rshunt_n3 n3 0 1e+09

.op
.end

* Auto-generated SPICE netlist (NO TEXT OCR USED)

M1 0 n4 0 0 NMOSdefault
E1 n2 0 n3 0 100k
I1 n1 0 DC 1m
* SAME_NODE_SKIPPED Capacitor both_on=0
.model NMOSdefault NMOS

* --- design-intent repair (does not change topology) ---
Rshunt_n1 n1 0 1e+09
Rshunt_n2 n2 0 1e+09
Rshunt_n4 n4 0 1e+09

.op
.end

* Auto-generated SPICE netlist (NO TEXT OCR USED)

* SAME_NODE_SKIPPED Diode both_on=n1
* SAME_NODE_SKIPPED Diode both_on=n1
E1 n4 0 n3 n1 100k
* SAME_NODE_SKIPPED Inductor both_on=n1
* SAME_NODE_SKIPPED Capacitor both_on=n1
E2 n1 0 n1 n2 100k

* --- design-intent repair (does not change topology) ---
Rshunt_n1 n1 0 1e+09

.op
.end

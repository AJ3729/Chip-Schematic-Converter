* Auto-generated SPICE netlist (NO TEXT OCR USED)

* SAME_NODE_SKIPPED Resistor both_on=0
C1 0 n2 1u
I1 0 n1 AC 1m
* SAME_NODE_SKIPPED Capacitor both_on=n1

* --- design-intent repair (does not change topology) ---
Rshunt_n1 n1 0 1e+09
Rshunt_n2 n2 0 1e+09

.op
.end

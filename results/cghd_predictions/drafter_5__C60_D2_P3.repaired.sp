* Auto-generated SPICE netlist (NO TEXT OCR USED)

E1 0 0 0 0 100k
I1 n3 n4 DC 1m
* SAME_NODE_SKIPPED Capacitor both_on=0
I2 n4 n2 DC 1m
* SAME_NODE_SKIPPED Resistor both_on=n1

* --- design-intent repair (does not change topology) ---
Rshunt_n1 n1 0 1e+09
Rshunt_n2 n2 0 1e+09

.op
.end

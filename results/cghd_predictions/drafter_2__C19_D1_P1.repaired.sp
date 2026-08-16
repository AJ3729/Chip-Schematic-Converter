* Auto-generated SPICE netlist (NO TEXT OCR USED)

V1 n2 0 DC 5
* UNSNAPPED Zener Diode raw_nodes=[0, None]
* SAME_NODE_SKIPPED Capacitor both_on=0
* SAME_NODE_SKIPPED Resistor both_on=0
* SAME_NODE_SKIPPED Capacitor both_on=0
E1 n3 0 0 n1 100k
* SAME_NODE_SKIPPED Capacitor both_on=0
* SAME_NODE_SKIPPED Capacitor both_on=0

* --- design-intent repair (does not change topology) ---
Rshunt_n1 n1 0 1e+09
Rshunt_n2 n2 0 1e+09

.op
.end

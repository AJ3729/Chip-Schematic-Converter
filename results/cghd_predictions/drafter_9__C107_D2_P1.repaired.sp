* Auto-generated SPICE netlist (NO TEXT OCR USED)

* SAME_NODE_SKIPPED Capacitor both_on=0
* SAME_NODE_SKIPPED Resistor both_on=0
R1 n1 0 1k
L1 0 n2 1m
* SAME_NODE_SKIPPED Inductor both_on=0
* SAME_NODE_SKIPPED Capacitor both_on=0
* SAME_NODE_SKIPPED Inductor both_on=0
* SAME_NODE_SKIPPED Inductor both_on=0
* SAME_NODE_SKIPPED Inductor both_on=0

* --- design-intent repair (does not change topology) ---
Rshunt_n1 n1 0 1e+09
Rshunt_n2 n2 0 1e+09

.op
.end

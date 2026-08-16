* Auto-generated SPICE netlist (NO TEXT OCR USED)

* SAME_NODE_SKIPPED Resistor both_on=0
* SAME_NODE_SKIPPED Resistor both_on=0
R1 n2 0 1k
R2 0 n3 1k
R3 0 n3 1k
R4 n1 0 1k
* SAME_NODE_SKIPPED Capacitor both_on=0
R5 0 n3 1k

* --- design-intent repair (does not change topology) ---
Rshunt_n1 n1 0 1e+09
Rshunt_n2 n2 0 1e+09

.op
.end

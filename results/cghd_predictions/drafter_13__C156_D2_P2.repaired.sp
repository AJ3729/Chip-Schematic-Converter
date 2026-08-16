* Auto-generated SPICE netlist (NO TEXT OCR USED)

I1 0 n1 AC 1m
I2 n1 0 AC 1m
I3 n2 0 AC 1m
C1 n1 0 1u
* SAME_NODE_SKIPPED Resistor both_on=n2

* --- design-intent repair (does not change topology) ---
Rshunt_n1 n1 0 1e+09
Rshunt_n2 n2 0 1e+09

.op
.end

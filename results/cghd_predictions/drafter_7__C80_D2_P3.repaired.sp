* Auto-generated SPICE netlist (NO TEXT OCR USED)

R1 n1 0 1k
* SAME_NODE_SKIPPED Resistor both_on=0
R2 n2 0 1k
* SAME_NODE_SKIPPED Resistor both_on=0
R3 n3 n5 1k
* SAME_NODE_SKIPPED Capacitor both_on=0
C1 n4 n5 1u
C2 n3 n5 1u

* --- design-intent repair (does not change topology) ---
Rshunt_n1 n1 0 1e+09
Rshunt_n2 n2 0 1e+09
Rshunt_n3 n3 0 1e+09

.op
.end

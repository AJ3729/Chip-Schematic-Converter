* Auto-generated SPICE netlist (NO TEXT OCR USED)

R1 n3 n4 1k
C1 n3 n4 1u
R2 0 n5 1k
R3 n2 0 1k
R4 n1 0 1k
* SAME_NODE_SKIPPED Resistor both_on=0
* SAME_NODE_SKIPPED Capacitor both_on=0

* --- design-intent repair (does not change topology) ---
Rshunt_n1 n1 0 1e+09
Rshunt_n2 n2 0 1e+09
Rshunt_n3 n3 0 1e+09
Rshunt_n5 n5 0 1e+09

.op
.end

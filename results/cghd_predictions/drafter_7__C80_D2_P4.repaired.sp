* Auto-generated SPICE netlist (NO TEXT OCR USED)

R1 0 n4 1k
R2 n2 0 1k
R3 n3 0 1k
* SAME_NODE_SKIPPED Resistor both_on=n1
* SAME_NODE_SKIPPED Resistor both_on=0
C1 0 n2 1u
C2 0 n2 1u

* --- design-intent repair (does not change topology) ---
Rshunt_n1 n1 0 1e+09
Rshunt_n3 n3 0 1e+09
Rshunt_n4 n4 0 1e+09

.op
.end

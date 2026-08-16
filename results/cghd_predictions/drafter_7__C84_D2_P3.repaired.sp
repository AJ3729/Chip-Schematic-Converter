* Auto-generated SPICE netlist (NO TEXT OCR USED)

* SAME_NODE_SKIPPED Resistor both_on=0
R1 n1 n2 1k
R2 0 n2 1k
R3 0 n1 1k
R4 n2 0 1k
I1 0 n2 DC 1m
C1 0 n3 1u

* --- design-intent repair (does not change topology) ---
Rshunt_n3 n3 0 1e+09

.op
.end

* Auto-generated SPICE netlist (NO TEXT OCR USED)

C1 n1 0 1u
* SAME_NODE_SKIPPED Inductor both_on=0
E1 0 0 n1 n2 100k
* SAME_NODE_SKIPPED Diode both_on=0

* --- design-intent repair (does not change topology) ---
Rshunt_n2 n2 0 1e+09

.op
.end

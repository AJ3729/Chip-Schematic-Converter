* Auto-generated SPICE netlist (NO TEXT OCR USED)

C1 0 n1 1u
* SAME_NODE_SKIPPED V-DC (one port) both_on=0
V1 n2 0 DC 5

* --- design-intent repair (does not change topology) ---
Rshunt_n1 n1 0 1e+09
Rshunt_n2 n2 0 1e+09

.op
.end

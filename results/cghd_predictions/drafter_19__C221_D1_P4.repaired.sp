* Auto-generated SPICE netlist (NO TEXT OCR USED)

R1 n1 0 1k
* SAME_NODE_SKIPPED Resistor both_on=0
D1 n2 0 Zdefault
D2 0 n2 Zdefault
D3 n3 0 Zdefault
.model Zdefault D(bv=5.1)

* --- design-intent repair (does not change topology) ---
Rshunt_n1 n1 0 1e+09
Rshunt_n3 n3 0 1e+09

.op
.end

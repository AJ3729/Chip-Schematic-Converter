* Auto-generated SPICE netlist (NO TEXT OCR USED)

V1 0 n4 DC 5
R1 n3 0 1k
* SAME_NODE_SKIPPED Resistor both_on=0
R2 n3 0 1k
D1 n1 0 Zdefault
C1 0 n3 1u
E1 n2 0 0 0 100k
.model Zdefault D(bv=5.1)

* --- design-intent repair (does not change topology) ---
Rshunt_n1 n1 0 1e+09
Rshunt_n2 n2 0 1e+09
Rshunt_n4 n4 0 1e+09

.op
.end

* Auto-generated SPICE netlist (NO TEXT OCR USED)

E1 n5 0 n6 n4 100k
* SAME_NODE_SKIPPED Resistor both_on=0
R1 n6 0 1k
R2 0 n4 1k
E2 n7 0 0 0 100k
R3 n4 0 1k
D1 n1 n3 Zdefault
R4 n1 0 1k
L1 0 n2 1m
.model Zdefault D(bv=5.1)

* --- design-intent repair (does not change topology) ---
Rshunt_n2 n2 0 1e+09
Rshunt_n3 n3 0 1e+09
Rshunt_n5 n5 0 1e+09
Rshunt_n7 n7 0 1e+09

.op
.end

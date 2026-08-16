* Auto-generated SPICE netlist (NO TEXT OCR USED)

R1 n1 0 1k
* SAME_NODE_SKIPPED Resistor both_on=0
R2 0 n3 1k
V1 n5 n4 DC 5
D1 n2 0 Ddefault
R3 0 n4 1k
V2 n5 n4 DC 5
R4 n2 0 1k
.model Ddefault D

* --- design-intent repair (does not change topology) ---
Rshunt_n1 n1 0 1e+09
Rshunt_n3 n3 0 1e+09

.op
.end

* Auto-generated SPICE netlist (NO TEXT OCR USED)

C1 n3 0 1u
R1 n4 0 1k
L1 n1 n2 1m
D1 n3 n4 Ddefault
.model Ddefault D

* --- design-intent repair (does not change topology) ---
Rshunt_n1 n1 0 1e+09

.op
.end

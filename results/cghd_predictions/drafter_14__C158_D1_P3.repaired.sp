* Auto-generated SPICE netlist (NO TEXT OCR USED)

D1 0 n3 Zdefault
C1 0 n5 1u
Q1 n4 0 n3 QPNPdefault
L1 n2 n1 1m
I1 n3 n5 DC 1m
.model QPNPdefault PNP
.model Zdefault D(bv=5.1)

* --- design-intent repair (does not change topology) ---
Rshunt_n1 n1 0 1e+09
Rshunt_n4 n4 0 1e+09
Rshunt_n5 n5 0 1e+09

.op
.end

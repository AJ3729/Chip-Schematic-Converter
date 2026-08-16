* Auto-generated SPICE netlist (NO TEXT OCR USED)

R1 0 n4 1k
D1 n3 0 Zdefault
C1 n1 n2 1u
V1 n1 0 DC 5
.model Zdefault D(bv=5.1)

* --- design-intent repair (does not change topology) ---
Rshunt_n1 n1 0 1e+09
Rshunt_n3 n3 0 1e+09
Rshunt_n4 n4 0 1e+09

.op
.end

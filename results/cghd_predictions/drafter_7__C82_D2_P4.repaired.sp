* Auto-generated SPICE netlist (NO TEXT OCR USED)

R1 0 n1 1k
D1 n4 n1 Zdefault
R2 n2 0 1k
D2 n3 n4 Zdefault
.model Zdefault D(bv=5.1)

* --- design-intent repair (does not change topology) ---
Rshunt_n2 n2 0 1e+09
Rshunt_n3 n3 0 1e+09

.op
.end

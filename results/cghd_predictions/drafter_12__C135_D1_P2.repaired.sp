* Auto-generated SPICE netlist (NO TEXT OCR USED)

I1 0 n2 DC 1m
R1 n1 0 1k
D1 0 n3 Zdefault
R2 0 n3 1k
R3 n3 0 1k
.model Zdefault D(bv=5.1)

* --- design-intent repair (does not change topology) ---
Rshunt_n1 n1 0 1e+09
Rshunt_n2 n2 0 1e+09

.op
.end

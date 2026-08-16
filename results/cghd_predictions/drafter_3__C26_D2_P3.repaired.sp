* Auto-generated SPICE netlist (NO TEXT OCR USED)

C1 0 n1 1u
R1 n1 0 1k
R2 n2 n3 1k
D1 n2 0 Zdefault
.model Zdefault D(bv=5.1)

* --- design-intent repair (does not change topology) ---
Rshunt_n3 n3 0 1e+09

.op
.end

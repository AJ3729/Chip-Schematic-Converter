* Auto-generated SPICE netlist (NO TEXT OCR USED)

V1 0 n2 AC 1
D1 0 n2 Zdefault
R1 n1 0 1k
C1 0 n2 1u
.model Zdefault D(bv=5.1)

* --- design-intent repair (does not change topology) ---
Rshunt_n1 n1 0 1e+09

.op
.end

* Auto-generated SPICE netlist (NO TEXT OCR USED)

R1 n2 0 1k
R2 n2 n3 1k
D1 0 n2 Zdefault
R3 n1 0 1k
D2 n1 0 Zdefault
D3 0 n2 Zdefault
.model Zdefault D(bv=5.1)

* --- design-intent repair (does not change topology) ---
Rshunt_n3 n3 0 1e+09

.op
.end

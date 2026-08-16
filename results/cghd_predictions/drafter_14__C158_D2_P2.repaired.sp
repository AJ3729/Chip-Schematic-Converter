* Auto-generated SPICE netlist (NO TEXT OCR USED)

C1 n2 0 1u
D1 n2 0 Zdefault
D2 n2 0 Zdefault
D3 0 n1 Ddefault
.model Ddefault D
.model Zdefault D(bv=5.1)

* --- design-intent repair (does not change topology) ---
Rshunt_n1 n1 0 1e+09

.op
.end

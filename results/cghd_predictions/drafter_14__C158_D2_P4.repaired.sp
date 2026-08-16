* Auto-generated SPICE netlist (NO TEXT OCR USED)

D1 0 n2 Zdefault
D2 n1 0 Zdefault
C1 n1 n2 1u
C2 n3 0 1u
C3 n1 n3 1u
D3 0 n1 Ddefault
D4 0 n1 Ddefault
.model Ddefault D
.model Zdefault D(bv=5.1)

* --- design-intent repair (does not change topology) ---
Rshunt_n3 n3 0 1e+09

.op
.end

* Auto-generated SPICE netlist (NO TEXT OCR USED)

D1 0 n2 Ddefault
R1 0 n3 1k
R2 n2 n4 1k
C1 0 n3 1u
D2 n2 n3 Zdefault
D3 n3 n2 Ddefault
D4 0 n1 Zdefault
.model Ddefault D
.model Zdefault D(bv=5.1)

* --- design-intent repair (does not change topology) ---
Rshunt_n1 n1 0 1e+09
Rshunt_n4 n4 0 1e+09

.op
.end

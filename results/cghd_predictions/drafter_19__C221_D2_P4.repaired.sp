* Auto-generated SPICE netlist (NO TEXT OCR USED)

C1 n2 0 1u
R1 n3 n1 1k
R2 n1 n3 1k
D1 n3 n1 Ddefault
.model Ddefault D

* --- design-intent repair (does not change topology) ---
Rshunt_n1 n1 0 1e+09
Rshunt_n2 n2 0 1e+09

.op
.end

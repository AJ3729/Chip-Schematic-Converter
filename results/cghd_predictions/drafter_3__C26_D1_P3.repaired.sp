* Auto-generated SPICE netlist (NO TEXT OCR USED)

D1 n1 0 Ddefault
R1 0 n2 1k
R2 n1 0 1k
C1 n1 0 1u
.model Ddefault D

* --- design-intent repair (does not change topology) ---
Rshunt_n2 n2 0 1e+09

.op
.end

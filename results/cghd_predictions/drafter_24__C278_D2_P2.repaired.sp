* Auto-generated SPICE netlist (NO TEXT OCR USED)

Q1 n3 0 n1 QPNPdefault
C1 n2 n3 1u
D1 n2 n4 Ddefault
.model Ddefault D
.model QPNPdefault PNP

* --- design-intent repair (does not change topology) ---
Rshunt_n1 n1 0 1e+09

.op
.end

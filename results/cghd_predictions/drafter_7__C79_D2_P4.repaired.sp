* Auto-generated SPICE netlist (NO TEXT OCR USED)

I1 n3 n5 AC 1m
R1 0 n4 1k
R2 0 n5 1k
R3 0 n5 1k
R4 n3 n2 1k
R5 n4 n1 1k
C1 n2 0 1u
D1 n1 n4 Ddefault
.model Ddefault D

* --- design-intent repair (does not change topology) ---
Rshunt_n2 n2 0 1e+09

.op
.end

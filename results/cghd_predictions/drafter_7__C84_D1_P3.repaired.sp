* Auto-generated SPICE netlist (NO TEXT OCR USED)

R1 n1 n2 1k
R2 n2 n3 1k
R3 n1 0 1k
I1 n3 0 DC 1m
R4 0 n3 1k
R5 n3 0 1k
I2 0 n1 DC 1m
C1 n4 n5 1u
R6 n1 n4 1k
D1 n4 0 Ddefault
C2 0 n5 1u
.model Ddefault D

* --- design-intent repair (does not change topology) ---
Rshunt_n5 n5 0 1e+09

.op
.end

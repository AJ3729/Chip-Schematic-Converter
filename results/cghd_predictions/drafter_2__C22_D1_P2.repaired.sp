* Auto-generated SPICE netlist (NO TEXT OCR USED)

V1 0 n1 DC 5
R1 n1 n3 1k
C1 0 n6 1u
I1 n3 n7 DC 1m
C2 n5 n6 1u
R2 n1 n4 1k
R3 n6 0 1k
D1 n4 n5 Zdefault
D2 n5 n2 Ddefault
.model Ddefault D
.model Zdefault D(bv=5.1)

* --- design-intent repair (does not change topology) ---
Rshunt_n2 n2 0 1e+09
Rshunt_n7 n7 0 1e+09

.op
.end

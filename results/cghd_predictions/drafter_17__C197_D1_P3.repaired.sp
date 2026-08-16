* Auto-generated SPICE netlist (NO TEXT OCR USED)

C1 0 n6 1u
I1 n4 0 DC 1m
C2 n4 n5 1u
L1 n1 n2 1m
D1 0 n3 Zdefault
R1 n4 n5 1k
R2 n6 n5 1k
R3 0 n4 1k
C3 0 n5 1u
.model Zdefault D(bv=5.1)

* --- design-intent repair (does not change topology) ---
Rshunt_n1 n1 0 1e+09
Rshunt_n3 n3 0 1e+09

.op
.end

* Auto-generated SPICE netlist (NO TEXT OCR USED)

V1 0 n4 DC 5
C1 n2 n4 1u
R1 n4 0 1k
R2 n4 0 1k
C2 n3 0 1u
E1 n3 0 n1 n2 100k
D1 0 n3 Zdefault
R3 n2 0 1k
C3 n5 0 1u
.model Zdefault D(bv=5.1)

* --- design-intent repair (does not change topology) ---
Rshunt_n1 n1 0 1e+09
Rshunt_n5 n5 0 1e+09

.op
.end

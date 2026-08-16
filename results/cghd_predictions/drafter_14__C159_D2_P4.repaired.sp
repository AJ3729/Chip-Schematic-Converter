* Auto-generated SPICE netlist (NO TEXT OCR USED)

E1 n3 0 n9 n1 100k
E2 n1 0 n2 0 100k
C1 n3 n9 1u
I1 n10 n7 DC 1m
I2 n4 n5 AC 1m
I3 n5 n6 AC 1m
M1 n9 0 n8 n8 NMOSdefault
D1 0 n1 Zdefault
.model NMOSdefault NMOS
.model Zdefault D(bv=5.1)

* --- design-intent repair (does not change topology) ---
Rshunt_n10 n10 0 1e+09
Rshunt_n2 n2 0 1e+09
Rshunt_n4 n4 0 1e+09
Rshunt_n8 n8 0 1e+09

.op
.end

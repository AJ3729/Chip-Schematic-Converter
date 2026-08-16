* Auto-generated SPICE netlist (NO TEXT OCR USED)

I1 n1 n3 AC 1m
L1 0 n1 1m
L2 n1 n2 1m
M1 n6 n6 n7 n7 NMOSdefault
I2 n4 n3 DC 1m
C1 n2 n3 1u
M2 n1 n2 n5 n5 NMOSdefault
C2 0 n3 1u
D1 n2 n4 Ddefault
D2 n2 n6 Ddefault
.model Ddefault D
.model NMOSdefault NMOS

* --- design-intent repair (does not change topology) ---
Rshunt_n3 n3 0 1e+09
Rshunt_n5 n5 0 1e+09
Rshunt_n7 n7 0 1e+09

.op
.end

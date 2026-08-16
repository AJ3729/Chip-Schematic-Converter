* Auto-generated SPICE netlist (NO TEXT OCR USED)

L1 n4 n3 1m
M1 n1 n2 0 0 NMOSdefault
C1 0 n3 1u
M2 0 0 n5 n5 NMOSdefault
.model NMOSdefault NMOS

* --- design-intent repair (does not change topology) ---
Rshunt_n1 n1 0 1e+09
Rshunt_n3 n3 0 1e+09
Rshunt_n5 n5 0 1e+09

.op
.end

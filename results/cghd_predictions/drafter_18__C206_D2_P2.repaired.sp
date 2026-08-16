* Auto-generated SPICE netlist (NO TEXT OCR USED)

C1 n5 0 1u
* SAME_NODE_SKIPPED Capacitor both_on=0
D1 n1 0 Zdefault
D2 n3 0 Zdefault
D3 n4 n5 Ddefault
M1 0 n2 n2 n2 NMOSdefault
.model Ddefault D
.model NMOSdefault NMOS
.model Zdefault D(bv=5.1)

* --- design-intent repair (does not change topology) ---
Rshunt_n1 n1 0 1e+09
Rshunt_n3 n3 0 1e+09
Rshunt_n4 n4 0 1e+09

.op
.end

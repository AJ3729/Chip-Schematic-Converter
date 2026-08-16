* Auto-generated SPICE netlist (NO TEXT OCR USED)

I1 n8 n10 AC 1m
C1 n7 n1 1u
I2 n10 n11 DC 1m
I3 n6 n8 AC 1m
D1 n1 n3 Zdefault
E1 n7 0 n1 n9 100k
* UNSNAPPED V-AC raw_nodes=[2, None]
D2 n3 n1 Zdefault
E2 n2 0 n4 n1 100k
C2 n1 n7 1u
M1 n7 n1 n7 n7 NMOSdefault
V1 n1 0 AC 1
D3 n5 n1 Zdefault
.model NMOSdefault NMOS
.model Zdefault D(bv=5.1)

* --- design-intent repair (does not change topology) ---
Rshunt_n10 n10 0 1e+09
Rshunt_n2 n2 0 1e+09
Rshunt_n5 n5 0 1e+09
Rshunt_n9 n9 0 1e+09

.op
.end

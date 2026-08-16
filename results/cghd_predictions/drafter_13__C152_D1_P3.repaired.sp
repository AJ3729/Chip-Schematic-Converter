* Auto-generated SPICE netlist (NO TEXT OCR USED)

I1 n2 n4 AC 1m
L1 n2 n3 1m
L2 0 n2 1m
M1 n7 n3 n4 n4 NMOSdefault
M2 n3 n2 n8 n8 NMOSdefault
M3 n3 n5 n5 n5 PMOSdefault
C1 n4 0 1u
D1 n3 n6 Ddefault
M4 n9 n4 n6 n6 NMOSdefault
V1 n9 n3 DC 5
M5 n3 n3 n9 n9 NMOSdefault
R1 0 n1 1k
.model Ddefault D
.model NMOSdefault NMOS
.model PMOSdefault PMOS

* --- design-intent repair (does not change topology) ---
Rshunt_n1 n1 0 1e+09
Rshunt_n4 n4 0 1e+09
Rshunt_n8 n8 0 1e+09

.op
.end

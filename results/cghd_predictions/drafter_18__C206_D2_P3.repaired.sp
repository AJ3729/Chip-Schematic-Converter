* Auto-generated SPICE netlist (NO TEXT OCR USED)

D1 n2 0 Zdefault
C1 0 n4 1u
M1 0 n5 0 0 PMOSdefault
C2 0 n5 1u
D2 n1 0 Zdefault
C3 n5 0 1u
C4 0 n3 1u
V1 n1 0 DC 5
.model PMOSdefault PMOS
.model Zdefault D(bv=5.1)

* --- design-intent repair (does not change topology) ---
Rshunt_n2 n2 0 1e+09
Rshunt_n3 n3 0 1e+09
Rshunt_n4 n4 0 1e+09
Rshunt_n5 n5 0 1e+09

.op
.end

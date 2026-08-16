* Auto-generated SPICE netlist (NO TEXT OCR USED)

D1 n1 n2 Zdefault
M1 n3 0 n4 n4 PMOSdefault
C1 n1 0 1u
Q1 n2 n3 0 QNPNdefault
D2 n3 0 Zdefault
D3 n2 n1 Ddefault
.model Ddefault D
.model PMOSdefault PMOS
.model QNPNdefault NPN
.model Zdefault D(bv=5.1)

* --- design-intent repair (does not change topology) ---
Rshunt_n4 n4 0 1e+09

.op
.end

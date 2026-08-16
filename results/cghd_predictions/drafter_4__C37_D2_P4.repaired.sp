* Auto-generated SPICE netlist (NO TEXT OCR USED)

C1 0 n8 1u
R1 n5 n8 1k
R2 n8 n4 1k
R3 n1 0 1k
M1 0 n3 n6 n6 PMOSdefault
R4 n1 n4 1k
M2 n3 n7 n5 n5 PMOSdefault
R5 n1 n3 1k
D1 n4 0 Ddefault
C2 0 n4 1u
C3 n3 n6 1u
D2 n4 n2 Ddefault
.model Ddefault D
.model PMOSdefault PMOS

* --- design-intent repair (does not change topology) ---
Rshunt_n2 n2 0 1e+09
Rshunt_n7 n7 0 1e+09

.op
.end

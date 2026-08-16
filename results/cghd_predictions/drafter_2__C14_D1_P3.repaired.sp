* Auto-generated SPICE netlist (NO TEXT OCR USED)

D1 n5 n2 Ddefault
D2 n6 n2 Ddefault
* SAME_NODE_SKIPPED I-DC both_on=0
R1 n6 0 1k
V1 n2 0 DC 5
R2 n9 n7 1k
M1 n11 0 0 0 PMOSdefault
M2 0 0 0 0 PMOSdefault
R3 n7 n8 1k
I1 0 n10 DC 1m
V2 0 n10 DC 5
R4 0 n9 1k
M3 n3 n4 n1 n1 PMOSdefault
E1 0 0 n12 n11 100k
.model Ddefault D
.model PMOSdefault PMOS

* --- design-intent repair (does not change topology) ---
Rshunt_n1 n1 0 1e+09
Rshunt_n12 n12 0 1e+09
Rshunt_n5 n5 0 1e+09
Rshunt_n8 n8 0 1e+09

.op
.end

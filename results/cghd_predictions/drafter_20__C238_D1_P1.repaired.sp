* Auto-generated SPICE netlist (NO TEXT OCR USED)

E1 n1 0 n2 n4 100k
E2 n3 0 n4 n6 100k
E3 n4 0 n10 n4 100k
R1 n3 n2 1k
R2 n6 n1 1k
C1 n2 n1 1u
R3 n11 n5 1k
R4 n6 n3 1k
I1 n11 n5 AC 1m
E4 n7 0 n8 n1 100k
R5 n9 n12 1k
L1 n7 n5 1m
R6 n12 0 1k

* --- design-intent repair (does not change topology) ---
Rshunt_n1 n1 0 1e+09
Rshunt_n9 n9 0 1e+09

.op
.end

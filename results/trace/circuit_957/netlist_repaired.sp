* Auto-generated SPICE netlist (NO TEXT OCR USED)

E1 n8 0 n11 n9 100k
V1 n7 n4 DC 5
E2 n6 0 n12 n10 100k
E3 n3 0 0 n4 100k
E4 n1 0 n6 n2 100k
E5 n14 0 n16 n13 100k
R1 n14 n5 1k
R2 n6 n9 1k
L1 n10 n6 1m
R3 n3 n2 1k
I1 n10 n15 DC 1m
L2 n13 n14 1m
R4 n1 n5 1k
R5 n8 n5 1k
C1 n4 n3 1u
L3 n2 n1 1m
C2 n6 n13 1u
C3 n9 n8 1u

* --- design-intent repair (does not change topology) ---
Rshunt_n15 n15 0 1e+09

.op
.end

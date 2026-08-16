* Auto-generated SPICE netlist (NO TEXT OCR USED)

C1 n9 n11 1u
C2 n13 n4 1u
* SAME_NODE_SKIPPED Capacitor both_on=n4
E1 n4 0 0 n7 100k
C3 n5 n6 1u
C4 n4 n7 1u
R1 n15 n13 1k
C5 n10 n12 1u
R2 n5 n6 1k
R3 n16 n17 1k
R4 n4 n7 1k
C6 n7 n16 1u
R5 n8 n4 1k
R6 n12 n5 1k
C7 n3 n1 1u
E2 n9 0 n14 n5 100k
R7 n17 n18 1k
R8 n15 n19 1k
E3 n2 0 n4 n4 100k
C8 n11 n13 1u
R9 n18 n1 1k

* --- design-intent repair (does not change topology) ---
Rshunt_n1 n1 0 1e+09
Rshunt_n10 n10 0 1e+09
Rshunt_n2 n2 0 1e+09
Rshunt_n8 n8 0 1e+09

.op
.end

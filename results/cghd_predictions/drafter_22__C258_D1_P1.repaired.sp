* Auto-generated SPICE netlist (NO TEXT OCR USED)

E1 n6 0 n18 n7 100k
E2 n3 0 n17 n9 100k
C1 n5 n4 1u
* SAME_NODE_SKIPPED Capacitor both_on=n3
R1 n5 n4 1k
C2 n1 n2 1u
R2 n11 n5 1k
E3 n5 0 n12 n4 100k
C3 n6 n7 1u
C4 n8 n13 1u
C5 n4 n8 1u
C6 n1 n2 1u
R3 n6 n7 1k
C7 n13 n3 1u
* SAME_NODE_SKIPPED Resistor both_on=n3
C8 n3 n6 1u
C9 n10 n11 1u
R4 n10 n16 1k
* SAME_NODE_SKIPPED Capacitor both_on=n3
R5 n14 n13 1k
R6 n7 0 1k
* SAME_NODE_SKIPPED Capacitor both_on=n7
R7 n15 n19 1k

* --- design-intent repair (does not change topology) ---
Rshunt_n1 n1 0 1e+09
Rshunt_n10 n10 0 1e+09
Rshunt_n15 n15 0 1e+09

.op
.end

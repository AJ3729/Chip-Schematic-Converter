* Auto-generated SPICE netlist (NO TEXT OCR USED)

C1 n9 n5 1u
E1 n10 0 0 n13 100k
C2 n1 n2 1u
E2 n5 0 n22 n4 100k
C3 n6 n7 1u
C4 n1 n2 1u
C5 n7 n16 1u
C6 n18 n4 1u
C7 n10 n13 1u
R1 n12 n14 1k
R2 n10 n13 1k
C8 n16 n18 1u
R3 n13 n20 1k
R4 n4 n5 1k
R5 n16 n21 1k
R6 n6 n7 1k
C9 n11 n12 1u
R7 n15 n19 1k
* SAME_NODE_SKIPPED Resistor both_on=n7
L1 n1 n24 1m
E3 n3 0 n4 n5 100k
R8 n21 n23 1k
* SAME_NODE_SKIPPED Diode both_on=n1
E4 n7 0 n17 n8 100k
V1 n15 0 DC 5
R9 n8 n6 1k
R10 n4 n9 1k
C10 n5 n10 1u
R11 n14 n6 1k

* --- design-intent repair (does not change topology) ---
Rshunt_n1 n1 0 1e+09
Rshunt_n11 n11 0 1e+09
Rshunt_n15 n15 0 1e+09
Rshunt_n20 n20 0 1e+09

.op
.end

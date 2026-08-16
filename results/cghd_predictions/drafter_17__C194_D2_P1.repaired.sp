* Auto-generated SPICE netlist (NO TEXT OCR USED)

E1 n1 0 n10 n5 100k
C1 n1 n12 1u
R1 n7 n1 1k
R2 n3 n6 1k
* SAME_NODE_SKIPPED Resistor both_on=n1
R3 n2 n1 1k
R4 n5 n1 1k
R5 n9 n1 1k
C2 n1 0 1u
C3 n2 n4 1u
R6 n1 0 1k
R7 n5 0 1k
R8 n1 n8 1k
R9 n10 0 1k
C4 n7 n1 1u
R10 n13 n11 1k
C5 n11 0 1u
V1 n12 0 DC 5
R11 n2 n1 1k
V2 n7 0 DC 5
* SAME_NODE_SKIPPED V-DC (one port) both_on=0

* --- design-intent repair (does not change topology) ---
Rshunt_n11 n11 0 1e+09
Rshunt_n12 n12 0 1e+09
Rshunt_n3 n3 0 1e+09
Rshunt_n4 n4 0 1e+09
Rshunt_n8 n8 0 1e+09
Rshunt_n9 n9 0 1e+09

.op
.end

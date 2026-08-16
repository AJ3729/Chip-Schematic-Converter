* Auto-generated SPICE netlist (NO TEXT OCR USED)

E1 n5 0 0 n8 100k
C1 n3 n4 1u
R1 n7 0 1k
* SAME_NODE_SKIPPED Resistor both_on=0
R2 n5 0 1k
R3 n4 0 1k
R4 n3 0 1k
R5 0 n6 1k
C2 n4 0 1u
R6 n1 0 1k
R7 n5 n9 1k
C3 0 n10 1u
C4 0 n9 1u
R8 n8 n9 1k
R9 0 n9 1k
D1 0 n2 Zdefault
C5 0 n3 1u
D2 n1 0 Zdefault
.model Zdefault D(bv=5.1)

* --- design-intent repair (does not change topology) ---
Rshunt_n10 n10 0 1e+09
Rshunt_n2 n2 0 1e+09
Rshunt_n6 n6 0 1e+09
Rshunt_n7 n7 0 1e+09

.op
.end

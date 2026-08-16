* Auto-generated SPICE netlist (NO TEXT OCR USED)

I1 n5 n3 AC 1m
V1 n6 n2 DC 5
C1 n3 n4 1u
R1 n7 n8 1k
C2 0 n5 1u
C3 n8 n9 1u
R2 n5 0 1k
R3 n1 n5 1k
R4 n7 0 1k
* SAME_NODE_SKIPPED V-AC both_on=0
L1 n1 n2 1m

* --- design-intent repair (does not change topology) ---
Rshunt_n3 n3 0 1e+09
Rshunt_n6 n6 0 1e+09
Rshunt_n9 n9 0 1e+09

.op
.end

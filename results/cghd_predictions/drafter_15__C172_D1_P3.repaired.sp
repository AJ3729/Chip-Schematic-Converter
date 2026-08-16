* Auto-generated SPICE netlist (NO TEXT OCR USED)

* SAME_NODE_SKIPPED I-AC both_on=n8
R1 n3 0 1k
* UNSNAPPED Capacitor raw_nodes=[12, None]
R2 n1 n3 1k
C1 n6 0 1u
C2 n8 n9 1u
R3 n1 n8 1k
R4 n7 n8 1k
R5 n9 n8 1k
R6 n6 0 1k
I1 n4 n3 DC 1m
C3 n1 n3 1u
V1 n1 0 DC 5
C4 n5 n2 1u
R7 n1 n2 1k
V2 n1 0 DC 5
I2 n3 n4 AC 1m

* --- design-intent repair (does not change topology) ---
Rshunt_n4 n4 0 1e+09
Rshunt_n5 n5 0 1e+09
Rshunt_n7 n7 0 1e+09

.op
.end

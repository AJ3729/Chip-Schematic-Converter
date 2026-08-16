* Auto-generated SPICE netlist (NO TEXT OCR USED)

V1 0 n2 DC 5
R1 n3 0 1k
R2 n2 n3 1k
* UNSNAPPED V-AC raw_nodes=[None, None]
* SAME_NODE_SKIPPED I-AC both_on=n1
I1 n3 0 DC 1m
* UNSNAPPED I-DC raw_nodes=[None, None]

* --- design-intent repair (does not change topology) ---
Rshunt_n1 n1 0 1e+09

.op
.end

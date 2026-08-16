* Auto-generated SPICE netlist (NO TEXT OCR USED)

V1 0 n2 DC 5
E1 0 0 0 0 100k
I1 n1 0 DC 1m
* SAME_NODE_SKIPPED Diode both_on=0
* SAME_NODE_SKIPPED Zener Diode both_on=0
* SAME_NODE_SKIPPED Zener Diode both_on=0
L1 n4 0 1m
C1 0 n3 1u
I2 n3 0 DC 1m
* SAME_NODE_SKIPPED Inductor both_on=0
* SAME_NODE_SKIPPED Inductor both_on=0
* SAME_NODE_SKIPPED Inductor both_on=0

* --- design-intent repair (does not change topology) ---
Rshunt_n1 n1 0 1e+09
Rshunt_n2 n2 0 1e+09
Rshunt_n3 n3 0 1e+09
Rshunt_n4 n4 0 1e+09

.op
.end

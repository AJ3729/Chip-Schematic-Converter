* Auto-generated SPICE netlist (NO TEXT OCR USED)

* SAME_NODE_SKIPPED Capacitor both_on=0
C1 n4 0 1u
L1 n3 n4 1m
C2 0 n3 1u
R1 n4 0 1k
V1 n1 n2 AC 1
* SAME_NODE_SKIPPED Diode both_on=0
I1 n1 n2 DC 1m

* --- design-intent repair (does not change topology) ---
Rshunt_n1 n1 0 1e+09

.op
.end

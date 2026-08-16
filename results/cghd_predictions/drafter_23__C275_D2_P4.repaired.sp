* Auto-generated SPICE netlist (NO TEXT OCR USED)

* SAME_NODE_SKIPPED Capacitor both_on=n2
E1 0 0 n8 n2 100k
C1 n13 n2 1u
C2 n14 n2 1u
* SAME_NODE_SKIPPED Capacitor both_on=n2
C3 n2 n4 1u
* SAME_NODE_SKIPPED Capacitor both_on=n2
* SAME_NODE_SKIPPED Inductor both_on=n2
C4 n12 n2 1u
I1 n3 n10 AC 1m
L1 n5 n7 1m
* UNSNAPPED Resistor raw_nodes=[9, None]
E2 n2 0 n11 n11 100k
* SAME_NODE_SKIPPED Capacitor both_on=n2
L2 n1 n5 1m
C5 n8 n2 1u
* SAME_NODE_SKIPPED Zener Diode both_on=0
R1 n9 0 1k

* --- design-intent repair (does not change topology) ---
Rshunt_n1 n1 0 1e+09
Rshunt_n10 n10 0 1e+09
Rshunt_n12 n12 0 1e+09
Rshunt_n13 n13 0 1e+09
Rshunt_n14 n14 0 1e+09
Rshunt_n4 n4 0 1e+09
Rshunt_n6 n6 0 1e+09
Rshunt_n9 n9 0 1e+09

.op
.end

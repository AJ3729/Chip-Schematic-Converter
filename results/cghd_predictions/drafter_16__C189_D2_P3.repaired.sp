* Auto-generated SPICE netlist (NO TEXT OCR USED)

* SAME_NODE_SKIPPED Zener Diode both_on=0
* SAME_NODE_SKIPPED Resistor both_on=0
C1 n4 0 1u
R1 n2 0 1k
D1 n2 0 Zdefault
R2 n4 n5 1k
C2 0 n3 1u
R3 0 n3 1k
E1 n6 0 n6 0 100k
* SAME_NODE_SKIPPED Capacitor both_on=0
* UNSNAPPED Capacitor raw_nodes=[0, None]
.model Zdefault D(bv=5.1)

* --- design-intent repair (does not change topology) ---
Rshunt_n1 n1 0 1e+09
Rshunt_n4 n4 0 1e+09

.op
.end

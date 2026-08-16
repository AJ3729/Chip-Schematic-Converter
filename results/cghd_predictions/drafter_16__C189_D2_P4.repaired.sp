* Auto-generated SPICE netlist (NO TEXT OCR USED)

* SAME_NODE_SKIPPED Zener Diode both_on=n3
R1 n2 n3 1k
D1 n2 n3 Zdefault
E1 n3 0 n3 n1 100k
C1 0 n6 1u
R2 n7 n8 1k
C2 n3 n10 1u
R3 n3 n6 1k
R4 n1 n5 1k
* SAME_NODE_SKIPPED Resistor both_on=n3
R5 n4 n5 1k
C3 0 n5 1u
C4 n7 n3 1u
* SAME_NODE_SKIPPED Resistor both_on=0
* SAME_NODE_SKIPPED Capacitor both_on=n5
E2 n3 0 n11 n9 100k
.model Zdefault D(bv=5.1)

* --- design-intent repair (does not change topology) ---
Rshunt_n1 n1 0 1e+09

.op
.end

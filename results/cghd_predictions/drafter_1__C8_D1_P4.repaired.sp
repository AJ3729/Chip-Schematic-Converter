* Auto-generated SPICE netlist (NO TEXT OCR USED)

V1 n2 0 DC 5
R1 0 n1 1k
* SAME_NODE_SKIPPED Resistor both_on=0
* SAME_NODE_SKIPPED Inductor both_on=0
* SAME_NODE_SKIPPED Capacitor both_on=0
* SAME_NODE_SKIPPED Diode both_on=0
* SAME_NODE_SKIPPED Capacitor both_on=0
* SAME_NODE_SKIPPED Capacitor both_on=0
D1 n1 0 Zdefault
.model Zdefault D(bv=5.1)

* --- design-intent repair (does not change topology) ---
Rshunt_n2 n2 0 1e+09

.op
.end

* Auto-generated SPICE netlist (NO TEXT OCR USED)

D1 n1 0 Zdefault
C1 n1 0 1u
* SAME_NODE_SKIPPED Resistor both_on=n2
Q1 0 n1 0 QNPNdefault
.model QNPNdefault NPN
.model Zdefault D(bv=5.1)

* --- design-intent repair (does not change topology) ---
Rshunt_n2 n2 0 1e+09

.op
.end

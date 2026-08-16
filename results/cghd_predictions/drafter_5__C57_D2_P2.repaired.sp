* Auto-generated SPICE netlist (NO TEXT OCR USED)

* SAME_NODE_SKIPPED Capacitor both_on=0
Q1 0 0 n3 QNPNdefault
C1 n2 0 1u
Q2 0 n2 n3 QNPNdefault
* SAME_NODE_SKIPPED Inductor both_on=n1
.model QNPNdefault NPN

* --- design-intent repair (does not change topology) ---
Rshunt_n1 n1 0 1e+09
Rshunt_n2 n2 0 1e+09

.op
.end

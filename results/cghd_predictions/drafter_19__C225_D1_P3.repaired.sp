* Auto-generated SPICE netlist (NO TEXT OCR USED)

E1 0 0 n5 n4 100k
E2 0 0 n1 n3 100k
M1 0 0 0 0 NMOSdefault
* SAME_NODE_SKIPPED Capacitor both_on=0
* SAME_NODE_SKIPPED Resistor both_on=0
* SAME_NODE_SKIPPED Capacitor both_on=0
E3 0 0 0 n2 100k
R1 0 n5 1k
* SAME_NODE_SKIPPED Capacitor both_on=0
.model NMOSdefault NMOS

* --- design-intent repair (does not change topology) ---
Rshunt_n1 n1 0 1e+09
Rshunt_n2 n2 0 1e+09
Rshunt_n4 n4 0 1e+09

.op
.end

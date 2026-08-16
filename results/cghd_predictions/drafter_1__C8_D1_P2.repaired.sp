* Auto-generated SPICE netlist (NO TEXT OCR USED)

R1 0 n3 1k
R2 0 n4 1k
D1 n4 0 Ddefault
* SAME_NODE_SKIPPED Resistor both_on=0
* SAME_NODE_SKIPPED Capacitor both_on=0
R3 n2 0 1k
V1 n2 n1 DC 5
D2 n4 0 Ddefault
V2 n2 n1 DC 5
* SAME_NODE_SKIPPED Zener Diode both_on=0
* SAME_NODE_SKIPPED Capacitor both_on=0
.model Ddefault D

* --- design-intent repair (does not change topology) ---
Rshunt_n3 n3 0 1e+09

.op
.end

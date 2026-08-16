* Auto-generated SPICE netlist (NO TEXT OCR USED)

C1 0 n1 1u
C2 0 n2 1u
C3 n2 0 1u
* SAME_NODE_SKIPPED Capacitor both_on=n1
V1 0 n1 AC 1
V2 n1 0 DC 5
C4 n1 0 1u

* --- design-intent repair (does not change topology) ---
Rshunt_n2 n2 0 1e+09

.op
.end

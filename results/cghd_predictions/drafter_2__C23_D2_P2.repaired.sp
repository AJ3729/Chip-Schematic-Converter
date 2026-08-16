* Auto-generated SPICE netlist (NO TEXT OCR USED)

V1 0 n2 DC 5
C1 n6 0 1u
* SAME_NODE_SKIPPED Zener Diode both_on=0
C2 n5 n7 1u
R1 n1 n5 1k
* SAME_NODE_SKIPPED Capacitor both_on=0
I1 n5 n6 DC 1m
R2 n1 n4 1k
R3 n1 0 1k
R4 n1 0 1k
R5 n3 0 1k
R6 n1 0 1k
D1 0 n6 Ddefault
D2 0 n7 Ddefault
C3 n7 0 1u
.model Ddefault D

* --- design-intent repair (does not change topology) ---
Rshunt_n2 n2 0 1e+09
Rshunt_n3 n3 0 1e+09
Rshunt_n4 n4 0 1e+09

.op
.end

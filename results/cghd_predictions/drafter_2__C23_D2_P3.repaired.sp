* Auto-generated SPICE netlist (NO TEXT OCR USED)

V1 n5 n1 DC 5
C1 n3 0 1u
* SAME_NODE_SKIPPED Capacitor both_on=0
C2 n2 0 1u
R1 n1 n3 1k
R2 n1 0 1k
R3 n1 n2 1k
R4 n1 0 1k
R5 n1 0 1k
R6 n1 0 1k
D1 n6 0 Zdefault
D2 n4 0 Ddefault
D3 0 n4 Ddefault
D4 n2 0 Zdefault
D5 0 n2 Ddefault
I1 n3 n2 DC 1m
.model Ddefault D
.model Zdefault D(bv=5.1)

* --- design-intent repair (does not change topology) ---
Rshunt_n5 n5 0 1e+09
Rshunt_n6 n6 0 1e+09

.op
.end

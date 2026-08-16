* Auto-generated SPICE netlist (NO TEXT OCR USED)

E1 n19 0 n13 n16 100k
E2 n14 0 n13 n11 100k
E3 n7 0 0 0 100k
C1 0 n6 1u
C2 n18 n14 1u
C3 n14 n13 1u
C4 n14 n15 1u
C5 n19 n13 1u
* SAME_NODE_SKIPPED Capacitor both_on=n13
C6 n9 0 1u
* SAME_NODE_SKIPPED V-DC (one port) both_on=0
I1 n12 n13 AC 1m
D1 n7 n4 Ddefault
C7 0 n16 1u
C8 n8 0 1u
Q1 0 n7 n10 QNPNdefault
R1 n15 0 1k
C9 n5 0 1u
* SAME_NODE_SKIPPED V-DC (one port) both_on=0
C10 n16 n17 1u
Q2 n4 n4 n5 QNPNdefault
V1 n4 0 DC 5
C11 n20 n19 1u
L1 0 n3 1m
L2 n2 0 1m
.model Ddefault D
.model QNPNdefault NPN

* --- design-intent repair (does not change topology) ---
Rshunt_n1 n1 0 1e+09
Rshunt_n10 n10 0 1e+09
Rshunt_n11 n11 0 1e+09
Rshunt_n2 n2 0 1e+09
Rshunt_n21 n21 0 1e+09
Rshunt_n3 n3 0 1e+09
Rshunt_n6 n6 0 1e+09
Rshunt_n8 n8 0 1e+09
Rshunt_n9 n9 0 1e+09

.op
.end

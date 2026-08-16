* Auto-generated SPICE netlist (NO TEXT OCR USED)

C1 n10 n11 1u
R1 n11 0 1k
R2 n16 n12 1k
C2 0 n9 1u
C3 n1 n4 1u
C4 n12 n1 1u
C5 n4 n14 1u
E1 n4 0 n1 n7 100k
C6 0 n3 1u
* SAME_NODE_SKIPPED Capacitor both_on=n1
R3 n6 n7 1k
R4 n14 n15 1k
C7 n9 n12 1u
R5 n5 n1 1k
R6 n17 n1 1k
R7 n16 n21 1k
Q1 n1 n18 n8 QNPNdefault
R8 n13 n20 1k
L1 n10 n13 1m
M1 0 n1 n2 n2 PMOSdefault
* SAME_NODE_SKIPPED Capacitor both_on=n1
.model PMOSdefault PMOS
.model QNPNdefault NPN

* --- design-intent repair (does not change topology) ---
Rshunt_n1 n1 0 1e+09
Rshunt_n10 n10 0 1e+09
Rshunt_n19 n19 0 1e+09
Rshunt_n3 n3 0 1e+09

.op
.end

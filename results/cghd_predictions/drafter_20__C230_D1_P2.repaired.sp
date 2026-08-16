* Auto-generated SPICE netlist (NO TEXT OCR USED)

Q1 n7 n9 n12 QNPNdefault
C1 n4 n5 1u
R1 n15 n9 1k
R2 n4 n5 1k
R3 n3 n4 1k
R4 n7 0 1k
I1 n9 n6 DC 1m
C2 n13 n9 1u
C3 n8 n6 1u
R5 n12 n9 1k
R6 0 n8 1k
Q2 n3 n4 0 QNPNdefault
R7 0 n7 1k
Q3 0 n7 n9 QPNPdefault
R8 0 n9 1k
R9 n12 n9 1k
R10 n11 n9 1k
R11 n14 n9 1k
V1 n9 0 DC 5
C4 n3 n2 1u
* SAME_NODE_SKIPPED Capacitor both_on=0
C5 0 n11 1u
I2 n13 n9 DC 1m
D1 n3 n1 Ddefault
V2 n2 0 DC 5
C6 n1 n2 1u
.model Ddefault D
.model QNPNdefault NPN
.model QPNPdefault PNP

* --- design-intent repair (does not change topology) ---
Rshunt_n10 n10 0 1e+09
Rshunt_n13 n13 0 1e+09
Rshunt_n14 n14 0 1e+09
Rshunt_n15 n15 0 1e+09
Rshunt_n2 n2 0 1e+09
Rshunt_n6 n6 0 1e+09

.op
.end

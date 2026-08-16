* Auto-generated SPICE netlist (NO TEXT OCR USED)

Q1 n1 n3 0 QNPNdefault
R1 n9 n14 1k
R2 0 n6 1k
R3 0 n8 1k
C1 n3 n4 1u
C2 n1 n2 1u
R4 n5 0 1k
C3 0 n11 1u
R5 n13 n6 1k
R6 n3 n4 1k
R7 n12 n6 1k
C4 n10 n6 1u
C5 n1 n2 1u
R8 n1 n3 1k
Q2 n6 n5 0 QPNPdefault
R9 0 n5 1k
R10 n9 n6 1k
R11 n11 n6 1k
I1 n10 n6 DC 1m
Q3 n5 n6 n9 QNPNdefault
* SAME_NODE_SKIPPED Capacitor both_on=0
V1 n6 0 DC 5
I2 n6 n7 DC 1m
C6 n14 n6 1u
C7 n8 n7 1u
V2 n7 n6 AC 1
Q4 n5 n6 n9 QNPNdefault
.model QNPNdefault NPN
.model QPNPdefault PNP

* --- design-intent repair (does not change topology) ---
Rshunt_n10 n10 0 1e+09
Rshunt_n12 n12 0 1e+09
Rshunt_n13 n13 0 1e+09
Rshunt_n2 n2 0 1e+09

.op
.end

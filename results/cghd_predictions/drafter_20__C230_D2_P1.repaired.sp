* Auto-generated SPICE netlist (NO TEXT OCR USED)

V1 0 n11 AC 1
Q1 n1 n5 n4 QNPNdefault
R1 n7 n4 1k
Q2 n7 0 n9 QNPNdefault
Q3 n4 n7 0 QPNPdefault
R2 n5 n6 1k
R3 n4 n8 1k
* SAME_NODE_SKIPPED Resistor both_on=0
R4 n10 0 1k
C1 n1 n2 1u
R5 n4 0 1k
R6 n1 n5 1k
R7 n9 0 1k
* SAME_NODE_SKIPPED Resistor both_on=0
R8 n4 n7 1k
C2 n4 n3 1u
R9 n9 0 1k
C3 n4 n3 1u
C4 n4 n10 1u
C5 n5 n6 1u
C6 n11 0 1u
I1 0 n8 DC 1m
C7 n1 n2 1u
.model QNPNdefault NPN
.model QPNPdefault PNP

* --- design-intent repair (does not change topology) ---
Rshunt_n2 n2 0 1e+09
Rshunt_n3 n3 0 1e+09

.op
.end

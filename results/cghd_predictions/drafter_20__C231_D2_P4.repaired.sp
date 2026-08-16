* Auto-generated SPICE netlist (NO TEXT OCR USED)

Q1 n6 n6 n8 QNPNdefault
C1 n4 n3 1u
Q2 n1 n7 n5 QNPNdefault
R1 n1 n7 1k
* SAME_NODE_SKIPPED Resistor both_on=n1
R2 n10 n8 1k
R3 n8 n7 1k
R4 n1 n6 1k
R5 n1 n2 1k
Q3 n1 n1 n1 QPNPdefault
Q4 n9 n8 n8 QPNPdefault
Q5 n1 n6 n11 QNPNdefault
R6 n4 n8 1k
R7 n11 n8 1k
Q6 n1 n4 n8 QNPNdefault
* SAME_NODE_SKIPPED Capacitor both_on=n1
Q7 n1 n1 n5 QNPNdefault
Q8 n5 n6 n8 QNPNdefault
Q9 n2 n3 n3 QPNPdefault
Q10 n4 n3 n1 QPNPdefault
Q11 n3 n1 n10 QNPNdefault
.model QNPNdefault NPN
.model QPNPdefault PNP

* --- design-intent repair (does not change topology) ---
Rshunt_n1 n1 0 1e+09

.op
.end

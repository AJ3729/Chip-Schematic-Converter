* Auto-generated SPICE netlist (NO TEXT OCR USED)

Q1 0 0 n3 QPNPdefault
R1 n2 0 1k
R2 n5 n6 1k
R3 0 n6 1k
R4 n1 n3 1k
C1 n2 0 1u
R5 n1 n2 1k
Q2 n2 n4 n5 QNPNdefault
.model QNPNdefault NPN
.model QPNPdefault PNP

* --- design-intent repair (does not change topology) ---
Rshunt_n4 n4 0 1e+09

.op
.end
